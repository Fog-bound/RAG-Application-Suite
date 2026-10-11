# api_server.py —— 把RAG包装成API服务
# 启动：uvicorn api_server:app --reload --port 8000
# 测试：POST http://127.0.0.1:8000/ask  Body: {"query": "你的问题"}
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 导入RAG主程序的检索+生成回答函数
from rag_main import (
    search_with_rerank,
    generate_answer,
)

app = FastAPI(title="RAG知识库问答API")

# 允许跨域（网页调用需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义请求格式
class Question(BaseModel):
    query: str  # 用户的问题

# ========== API接口1：问答 ==========
@app.post("/ask")
def ask(question: Question):
    query = question.query
    print(f"\n收到问题：{query}")

    # 1. 检索（粗排+精排）
    top_docs = search_with_rerank(query, recall_k=50, rerank_top_n=10)

    # 2. 生成回答
    if top_docs:
        answer = generate_answer(query, top_docs)
        return {
            "question": query,
            "answer": answer,
            "sources": [{"file": d["file_name"], "text": d["chunk_text"][:200]} for d in top_docs]
        }
    else:
        return {
            "question": query,
            "answer": "未找到相关文档，无法回答。",
            "sources": []
        }

# ========== API接口2：健康检查 ==========
@app.get("/")
def health_check():
    return {"status": "running", "message": "RAG服务已启动"}
