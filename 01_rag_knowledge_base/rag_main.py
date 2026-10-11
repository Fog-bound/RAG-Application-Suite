import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pathlib import Path
from typing import List
import pdfplumber
from minio import Minio
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, DataType, utility
from pymilvus import CollectionSchema
import requests
from openai import OpenAI

import config  # 统一配置（API Key 从这里读，不要写死在代码里）

# ======================== 【配置区域】 ========================
# Milvus 向量数据库路径（默认当前目录下）
MILVUS_DB_PATH = os.environ.get("MILVUS_DB_PATH", "./milvus-lite-data/milvus.db")
COLLECTION_NAME = "knowledge_base"

# MinIO 对象存储配置（请通过环境变量设置，不要硬编码密钥）
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = config.MINIO_SECRET_KEY
BUCKET_NAME = os.environ.get("BUCKET_NAME", "rag-files")

# BGE 中文Embedding模型
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBED_DIM = 512
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索："

# 文本滑动窗口切块参数
CHUNK_SIZE = 350
CHUNK_OVERLAP = 60

# 需要批量扫描导入的本地文档目录
DOC_ROOT_DIR = os.environ.get("DOC_ROOT_DIR", "./docs")
# 支持解析的文件后缀
SUPPORT_SUFFIX = {".pdf", ".txt", ".md"}
# ======================================================================

# ---------------------- 初始化 Milvus Lite 3.x ----------------------
db_file = Path(MILVUS_DB_PATH).resolve()
db_file.parent.mkdir(parents=True, exist_ok=True)
uri_path = db_file.as_posix()          # 移除 sqlite:/// 前缀
print(f" Milvus 数据库路径: {uri_path}")   # 调试输出
connections.connect(alias="default", uri=uri_path)

# ---------------------- 初始化 MinIO 客户端 ----------------------
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# ---------------------- 加载BGE向量模型 ----------------------
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# ---------------------- 初始化DeepSeek大模型（生成回答用） ----------------------
llm = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

# ---------------------- 初始化Milvus向量集合 ----------------------
def init_collection():
    if utility.has_collection(COLLECTION_NAME):
        coll = Collection(COLLECTION_NAME)
        coll.load()   #  确保加载
        print(f" 已加载已有集合【{COLLECTION_NAME}】")
        return coll

    # 以下是新建集合的代码（保持不变）
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="file_suffix", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM)
    ]
    schema = CollectionSchema(fields=fields, description="知识库向量集合")
    coll = Collection(name=COLLECTION_NAME, schema=schema)
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE", # "L2",
        "params": {"nlist": 128}
    }
    coll.create_index(field_name="vector", index_params=index_params)
    coll.load()   # 新建后也加载
    print(f" 向量集合【{COLLECTION_NAME}】创建完成")
    return coll


collection = init_collection()


# ---------------------- MinIO 工具函数 ----------------------
def ensure_bucket():
    """不存在则自动创建Bucket"""
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)
        print(f" MinIO Bucket【{BUCKET_NAME}】已创建")

def upload_to_minio(local_path: str, obj_name: str):
    """本地文件上传MinIO"""
    minio_client.fput_object(BUCKET_NAME, obj_name, local_path)
    return obj_name

def download_from_minio(obj_name: str, save_path: str):
    """从MinIO下载文件到本地临时路径"""
    minio_client.fget_object(BUCKET_NAME, obj_name, save_path)
    return save_path

# ---------------------- 文件解析：pdf / txt / md ----------------------
def extract_file_text(file_path: str) -> str:
    suffix = os.path.splitext(file_path)[1].lower()
    text = ""
    if suffix == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_txt = page.extract_text()
                if page_txt:
                    text += page_txt + "\n"
    elif suffix in (".txt", ".md", ".py"):
        # 优先utf-8，失败尝试gbk（Windows中文文档）
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="gbk") as f:
                text = f.read()
    # 简单清洗多余换行
    text = text.replace("\r", "").replace("\n\n", "\n").strip()
    return text

# ---------------------- 滑动窗口文本切块 ----------------------
def sliding_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    total_len = len(text)
    while start < total_len:
        end = min(start + chunk_size, total_len)
        seg = text[start:end].strip()
        if seg:
            chunks.append(seg)
        start += chunk_size - overlap
    return chunks

# ---------------------- 文本块向量化存入Milvus ----------------------
def save_chunks_to_milvus(chunks: List[str], filename: str, suffix: str):
    # BGE规范：文档片段不需要指令，查询语句需要指令
    vectors = embed_model.encode(chunks)
    data = [
        chunks,
        [filename] * len(chunks),
        [suffix] * len(chunks),
        vectors
    ]
    collection.insert(data)
    collection.flush()
    print(f" {filename} 成功入库 {len(chunks)} 个文本块")

# ---------------------- 【批量导入模块】遍历文件夹全链路 ----------------------
def import_folder_to_vector_db(root_dir: str):
    ensure_bucket()
    if not os.path.isdir(root_dir):
        print(f"目录不存在: {root_dir}")
        return

    for root, _, files in os.walk(root_dir):
        for fname in files:
            suffix = os.path.splitext(fname)[1].lower()
            if suffix not in SUPPORT_SUFFIX:
                continue
            local_file = os.path.join(root, fname)
            print(f"\n==== 正在处理文件：{local_file} ====")

            # 1. 原始文档上传MinIO持久化
            obj_key = fname
            upload_to_minio(local_file, obj_key)

            # 2. 从MinIO下载至临时文件用于文本解析
            temp_file = f"./.tmp_{fname}"
            download_from_minio(obj_key, temp_file)

            # 3. 提取文档文本
            full_text = extract_file_text(temp_file)
            if not full_text:
                print(f" 文件无有效文本，跳过：{fname}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                continue

            # 4. 文本滑动切块
            chunks = sliding_chunk(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

            # 5. Embedding并写入Milvus Lite
            save_chunks_to_milvus(chunks, fname, suffix)

            # 删除临时解析文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
    print("\n 文件夹内所有文档导入流程执行完毕！")

# ---------------------- 【检索模块】用户提问粗排召回 ----------------------
def search_knowledge(query: str, top_k=100):
    # BGE规范：查询必须拼接指令
    query_vec = embed_model.encode(f"{BGE_QUERY_INSTRUCTION}{query}")
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

    results = collection.search(
        data=[query_vec],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["chunk_text", "file_name", "file_suffix"]
    )

    print(f"\n==================== 用户问题：{query} ====================")
    for hits in results:
        for hit in hits:
            print(f"\n【距离分数】{hit.distance:.4f}")
            print(f"【文档名称】{hit.entity.get('file_name')}")
            print(f"【文件类型】{hit.entity.get('file_suffix')}")
            print(f"【文本片段】{hit.entity.get('chunk_text')}")
    return results

# ---------------------- 【重排模块】BGE-Reranker 精排 ----------------------
rerank_url = config.RERANK_URL

def rerank_documents(query: str, documents: List[str], top_n=10) -> List[int]:
    url = rerank_url
    print(f"Rerank 请求 URL: {url}")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {config.RERANK_API_KEY}"   # Key 从 config 读取
    }
    payload = {
        "model": config.RERANK_MODEL,
        "query": query,
        "documents": documents
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # 假设返回格式为 {"results": [{"index": 0, "relevance_score": 0.99}]}
        sorted_items = sorted(data["results"], key=lambda x: x["relevance_score"], reverse=True)[:top_n]
        return [item["index"] for item in sorted_items]
    except Exception as e:
        print(f"Rerank 失败，使用原始顺序: {e}")
        return list(range(min(top_n, len(documents))))

def search_with_rerank(query: str, recall_k=50, rerank_top_n=10):
    results = search_knowledge(query, top_k=recall_k)
    if not results or len(results[0]) == 0:
        print("未召回任何文档")
        return []
    candidates = []
    for hit in results[0]:
        candidates.append({
            "chunk_text": hit.entity.get('chunk_text'),
            "file_name": hit.entity.get('file_name'),
            "file_suffix": hit.entity.get('file_suffix'),
            "distance": hit.distance
        })
    doc_texts = [c["chunk_text"] for c in candidates]
    sorted_indices = rerank_documents(query, doc_texts, top_n=rerank_top_n)

    print(f"\n==================== 用户问题：{query} ====================")
    print(f"粗排召回 {len(candidates)} 条，重排序后展示前 {len(sorted_indices)} 条\n")
    for idx in sorted_indices:
        item = candidates[idx]
        print(f"【原始距离】{item['distance']:.4f}")
        print(f"【文档名称】{item['file_name']}")
        print(f"【文件类型】{item['file_suffix']}")
        print(f"【文本片段】{item['chunk_text']}")
        print("-" * 50)
    return [candidates[idx] for idx in sorted_indices]

# ---------------------- 【生成回答模块】把检索结果交给DeepSeek写答案 ----------------------
def build_prompt(query: str, context_docs: str) -> str:
    prompt = f"""你是一个拥有10年经验的知识库搜索总结专家，请根据以下提供的文档片段，回答用户的问题。
        如果文档中没有足够的信息，请明确告知用户"根据已有资料无法回答该问题"。

        用户问题：{query}

        提供的文档片段：
        {context_docs}

        请用简洁、清晰的风格回答用户问题，并尽可能引用文档中的信息。回答：

        ***如果文档中没有相关信息，请直接回答"根据已有资料无法回答该问题",不要胡编乱造。***
    """
    return prompt

def generate_answer(query: str, context_docs: List[dict]) -> str:
    """把精排选出的文档拼成上下文，交给DeepSeek生成最终回答"""
    context_text = ""
    for i, doc in enumerate(context_docs, 1):
        context_text += f"[文档{i}] {doc['chunk_text']}\n\n"

    prompt = build_prompt(query, context_text)
    try:
        response = llm.chat.completions.create(
            model=config.DEEPSEEK_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个基于文档的知识问答助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1024,
            top_p=0.9
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"生成回答时出错: {e}")
        return "抱歉，生成回答时出现错误，请稍后重试。"


if __name__ == "__main__":
    # ========== 运行模式切换 ==========
    # 模式1：批量扫描文件夹，文档入库（首次导入开启，导完注释掉，避免重复入库）
    # import_folder_to_vector_db(DOC_ROOT_DIR)

    # 模式2：问答（检索 → 重排 → 生成回答）
    query = "令狐冲的爱人是谁？"

    top_docs = search_with_rerank(query, recall_k=100, rerank_top_n=20)

    if top_docs:
        final_answer = generate_answer(query, top_docs)
        print("\n========== 最终回答 ==========")
        print(final_answer)
    else:
        print("未找到相关文档，无法回答。")
