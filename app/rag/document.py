"""
文档处理与向量化模块 (RAG)
负责：加载文档 → 分块 → 向量化 → 存入 ChromaDB → 检索
优化：单例缓存 + 批量向量化 + 文档更新去重 + 缓存失效
"""
import os
import time
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from app.config import settings


# ===== 单例缓存 =====
_vector_store = None
_embeddings = None

# ===== 文档列表缓存（避免每次从ChromaDB读取） =====
_doc_list_cache = {"data": None, "timestamp": 0}
_DOC_LIST_CACHE_TTL = 30  # 文档列表缓存30秒


def get_embeddings():
    """获取 Embedding 模型（单例缓存，避免重复创建）"""
    global _embeddings
    if _embeddings is None:
        embedding_model = getattr(settings, 'EMBEDDING_MODEL', 'embedding-3')
        _embeddings = OpenAIEmbeddings(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            model=embedding_model,
        )
    return _embeddings


def get_vector_store():
    """获取 ChromaDB 向量数据库实例（单例缓存，避免重复创建）"""
    global _vector_store
    if _vector_store is None:
        embeddings = get_embeddings()
        _vector_store = Chroma(
            persist_directory=settings.CHROMA_DIR,
            embedding_function=embeddings,
        )
    return _vector_store


def load_document(file_path: str) -> list:
    """
    根据文件类型加载文档
    支持：PDF、TXT、DOCX
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，仅支持 PDF/TXT/DOCX")

    return loader.load()


def read_document_content(file_path: str) -> str:
    """
    读取文档的纯文本内容（用于文档修改功能）

    Args:
        file_path: 文档路径

    Returns:
        str: 文档纯文本内容
    """
    docs = load_document(file_path)
    content = "\n\n".join([doc.page_content for doc in docs])
    return content


def split_documents(docs: list, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    """
    文档分块
    - chunk_size: 每块最大字符数
    - chunk_overlap: 块间重叠字符数（保证上下文连续性）
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    return splitter.split_documents(docs)


def index_document(file_path: str, filename: str = None) -> dict:
    """
    完整的文档索引流程：加载 → 分块 → 向量化 → 存储
    优化：如果文档已存在，先删除旧索引再重新索引，避免重复

    Returns:
        dict: 包含分块数量和状态信息
    """
    if filename is None:
        filename = os.path.basename(file_path)

    # 1. 加载文档
    docs = load_document(file_path)

    # 2. 给文档添加元数据
    for doc in docs:
        doc.metadata["source_file"] = filename

    # 3. 分块
    chunks = split_documents(docs)

    # 4. 如果文档已存在，先删除旧的索引（避免重复）
    vector_store = get_vector_store()
    try:
        collection = vector_store._collection
        # 删除同名的旧文档分块
        collection.delete(where={"source_file": filename})
    except Exception:
        pass  # 删除失败不影响后续索引

    # 5. 批量向量化并存储（使用add_documents一次性写入）
    vector_store.add_documents(chunks)

    # 6. 失效文档列表缓存
    _invalidate_doc_list_cache()

    return {
        "filename": filename,
        "chunks": len(chunks),
        "status": "success",
        "message": f"文档 {filename} 已成功索引，共 {len(chunks)} 个分块",
    }


def _invalidate_doc_list_cache():
    """失效文档列表缓存"""
    global _doc_list_cache
    _doc_list_cache = {"data": None, "timestamp": 0}


def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """
    在向量数据库中检索与查询最相关的文档片段

    Args:
        query: 用户查询
        top_k: 返回最相关的 K 个结果

    Returns:
        list[dict]: 检索结果列表
    """
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(query, k=top_k)

    formatted = []
    for doc, score in results:
        formatted.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source_file", "未知来源"),
            "relevance_score": round(1 - score, 4),  # 转换为相似度
        })

    return formatted


def list_indexed_documents() -> list[str]:
    """列出知识库中所有已索引的文档（带缓存，30秒TTL）"""
    global _doc_list_cache
    
    # 检查缓存是否有效
    now = time.time()
    if _doc_list_cache["data"] is not None and (now - _doc_list_cache["timestamp"]) < _DOC_LIST_CACHE_TTL:
        return _doc_list_cache["data"]
    
    vector_store = get_vector_store()
    # 从 ChromaDB 的元数据中提取所有文档名
    try:
        collection = vector_store._collection
        all_docs = collection.get(include=["metadatas"])
        sources = set()
        for meta in all_docs["metadatas"]:
            if meta and "source_file" in meta:
                sources.add(meta["source_file"])
        result = sorted(list(sources))
    except Exception:
        result = []
    
    # 更新缓存
    _doc_list_cache = {"data": result, "timestamp": now}
    return result


def delete_document(filename: str) -> dict:
    """
    从知识库中删除指定文档的索引和文件

    Args:
        filename: 文档文件名

    Returns:
        dict: 删除结果
    """
    vector_store = get_vector_store()
    try:
        collection = vector_store._collection
        # 删除该文档的所有分块
        collection.delete(where={"source_file": filename})
    except Exception as e:
        return {"status": "error", "message": f"删除索引失败: {str(e)}"}

    # 删除源文件
    file_path = os.path.join(settings.DOCUMENTS_DIR, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            return {"status": "warning", "message": f"索引已删除，但文件删除失败: {str(e)}"}

    # 失效缓存
    _invalidate_doc_list_cache()

    return {"status": "success", "message": f"文档 {filename} 已从知识库中删除"}


def update_document(file_path: str, filename: str = None) -> dict:
    """
    更新文档索引（先删旧索引，再重新索引）
    优化版：使用 delete + index_document 组合，确保去重

    Args:
        file_path: 文档路径
        filename: 文档名

    Returns:
        dict: 更新结果
    """
    if filename is None:
        filename = os.path.basename(file_path)

    # index_document 内部已实现先删旧索引再重建的逻辑
    return index_document(file_path, filename)


def get_document_content(filename: str) -> dict:
    """
    获取指定文档的文本内容

    Args:
        filename: 文档文件名

    Returns:
        dict: 包含文档内容的字典
    """
    file_path = os.path.join(settings.DOCUMENTS_DIR, filename)
    if not os.path.exists(file_path):
        # 也尝试直接用filename作为路径
        if os.path.exists(filename):
            file_path = filename
        else:
            return {"status": "error", "message": f"文件不存在: {filename}"}

    try:
        content = read_document_content(file_path)
        return {"status": "success", "content": content, "filename": filename}
    except Exception as e:
        return {"status": "error", "message": f"读取文档失败: {str(e)}"}


def export_document_as_docx(content: str, output_filename: str = None) -> dict:
    """
    将文本内容导出为DOCX格式文件

    Args:
        content: 文档文本内容
        output_filename: 输出文件名

    Returns:
        dict: 导出结果，包含下载路径
    """
    if output_filename is None:
        output_filename = "exported_document.docx"

    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    export_dir = os.path.join(static_dir, "modified")
    os.makedirs(export_dir, exist_ok=True)
    output_path = os.path.join(export_dir, output_filename)

    try:
        try:
            from docx import Document
            doc = Document()
            paragraphs = content.split("\n")
            for p_text in paragraphs:
                doc.add_paragraph(p_text)
            doc.save(output_path)
        except ImportError:
            # 没有python-docx就保存为txt
            output_filename = output_filename.replace(".docx", ".txt")
            output_path = os.path.join(export_dir, output_filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

        return {
            "status": "success",
            "message": f"文档导出成功",
            "download_url": f"/static/modified/{output_filename}",
            "filename": output_filename,
        }
    except Exception as e:
        return {"status": "error", "message": f"导出文档失败: {str(e)}"}
