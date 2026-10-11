# 通用文档 RAG 知识库问答系统

## 项目简介

基于 BGE 向量模型 + DeepSeek 大模型搭建的文档知识库问答系统，实现文档上传、向量化存储、语义检索、重排序、大模型生成回答的完整 RAG 流程，并通过 FastAPI 提供 HTTP 接口。

## 技术架构

```
文档上传 → PDF解析 → 文本分段 → BGE向量化 → Milvus存储
                                          ↓
用户提问 → BGE向量化 → Milvus检索 → BGE-Rerank重排序 → DeepSeek生成回答
```

## 核心功能

### 1. 文档处理
- 支持 PDF / TXT / MD 格式文档解析（pdfplumber，自动识别 utf-8/gbk 编码）
- 滑动窗口分段（chunk_size=350, overlap=60），保证语义完整

### 2. 向量检索（粗排）
- BGE-small-zh-v1.5 生成中文向量（512维，查询自动拼接 BGE 指令前缀）
- Milvus Lite 向量数据库存储
- 余弦相似度召回 TopK 相关片段

### 3. 重排序（精排）
- BGE-Reranker 对召回结果二次打分排序
- 提升检索精度，降低无关片段干扰

### 4. 对象存储
- MinIO 存储原始文档，支持批量扫描导入本地文档目录

### 5. 大模型生成回答
- DeepSeek 根据检索到的文档片段生成回答（temperature=0.3，稳定不瞎编）
- 系统提示词约束：无相关内容时明确告知"根据已有资料无法回答"，避免幻觉

### 6. HTTP 接口
- FastAPI 提供 POST /ask 问答接口，支持网页/Postman 调用

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置密钥（复制 .env.example 为 .env 并填写真实值）
#    需要填：DEEPSEEK_API_KEY（DeepSeek）、RERANK_API_KEY（XInference）、MINIO_SECRET_KEY
cp ../.env.example .env

# 3. 首次运行：导入文档（取消 rag_main.py 里 import_folder_to_vector_db 的注释）
python rag_main.py

# 4. 问答模式（恢复注释后运行）
python rag_main.py

# 5. 启动 API 服务
uvicorn api_server:app --reload --port 8000
# 测试：POST http://127.0.0.1:8000/ask  Body: {"query": "你的问题"}
```

## 配置说明

通过环境变量 / .env 配置（均有默认值）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| MILVUS_DB_PATH | ./milvus-lite-data/milvus.db | Milvus 数据库路径 |
| MINIO_ENDPOINT | localhost:9000 | MinIO 地址 |
| MINIO_ACCESS_KEY | admin | MinIO 用户名 |
| MINIO_SECRET_KEY | your-minio-secret-key | MinIO 密码（**请自行设置**） |
| BUCKET_NAME | rag-files | 存储桶名称 |
| DOC_ROOT_DIR | ./docs | 文档目录 |
| DEEPSEEK_API_KEY | （空） | DeepSeek API Key（**必填**） |
| DEEPSEEK_MODEL_NAME | deepseek-chat | DeepSeek 模型名 |
| RERANK_API_KEY | （空） | XInference Rerank API Key（**必填**） |
| RERANK_URL | http://localhost:9997/v1/rerank | Rerank 服务地址 |
| RERANK_MODEL | bge-reranker-v2-m3 | Rerank 模型名 |

## 项目结构

```
01_rag_knowledge_base/
├── rag_main.py          # 主程序（文档导入 + 检索 + 重排 + 生成回答）
├── api_server.py        # FastAPI HTTP 接口
├── config.py            # 统一配置（密钥从环境变量/.env 读取）
├── requirements.txt     # 依赖清单
└── README.md            # 本文件
```

## 技术栈

- **向量模型**：BGE-small-zh-v1.5（Embedding）/ BGE-Reranker（重排）
- **大模型**：DeepSeek（deepseek-chat）
- **向量数据库**：Milvus Lite
- **对象存储**：MinIO
- **文档解析**：pdfplumber
- **API 框架**：FastAPI + uvicorn
