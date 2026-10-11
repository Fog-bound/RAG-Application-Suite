# RAG Application Suite

大模型应用开发作品集，包含两个实战项目：RAG 知识库问答系统 + Dify 电商智能客服。

## 项目列表

| 项目 | 技术栈 | 说明 |
|------|--------|------|
| [01_rag_knowledge_base](./01_rag_knowledge_base) | Milvus, MinIO, BGE, DeepSeek, FastAPI | 通用文档 RAG 知识库问答系统 |
| [02_dify_customer_service](./02_dify_customer_service) | Dify, RAG, Agent工作流 | 基于 Dify 的电商智能客服系统 |

---

## 01. RAG 知识库问答系统

### 项目简介

基于 BGE 向量模型 + DeepSeek 大模型搭建的文档知识库问答系统，实现文档上传、向量化存储、语义检索、重排序、大模型生成回答的完整 RAG 流程，并通过 FastAPI 提供 HTTP 接口。

### 技术架构

```
文档上传 → PDF解析 → 文本分段 → BGE向量化 → Milvus存储
                                          ↓
用户提问 → BGE向量化 → Milvus检索 → BGE-Rerank重排序 → DeepSeek生成回答
```

### 核心功能

- **文档处理**：PDF/TXT/MD 文档解析，滑动窗口分段（chunk_size=350, overlap=60）
- **向量检索**：BGE-small-zh-v1.5 生成向量，Milvus Lite 存储，余弦相似度召回 TopK
- **重排序优化**：BGE-Reranker 二次排序，提升检索精度
- **对象存储**：MinIO 存储原始文档，支持批量导入
- **Prompt 工程**：系统提示词约束基于检索内容回答，无相关内容时明确告知，避免幻觉
- **HTTP 接口**：FastAPI 提供 POST /ask 问答接口

### 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置密钥（复制 .env.example 为 .env，填 DEEPSEEK_API_KEY / RERANK_API_KEY / MINIO_SECRET_KEY）
cp .env.example .env

# 3. 首次运行导入文档（取消 rag_main.py 里导入代码的注释）
python rag_main.py

# 4. 启动 API 服务
uvicorn api_server:app --reload --port 8000
```

### 项目结构

```
01_rag_knowledge_base/
├── rag_main.py          # 主程序（文档导入+检索+重排+生成回答）
├── api_server.py        # FastAPI HTTP 接口
├── config.py            # 统一配置（密钥从环境变量/.env 读取）
├── requirements.txt     # 依赖清单
└── README.md            # 详细说明
```

---

## 02. Dify 电商智能客服系统

### 项目简介

基于 Dify 低代码平台搭建的电商智能客服工作流，实现情绪识别、意图识别、问题分类路由、多知识库 RAG 检索、多轮对话记忆等功能。

### 技术架构

```
用户消息 → 情绪识别 → 意图识别 → 4分类路由（售后/物流/商品/其他）
                ↓
        多知识库 RAG 检索（BGE-Embedding + BGE-Rerank）
                ↓
        大模型生成回答 / 愤怒用户直接转人工
```

### 核心功能

- **工作流编排**：情绪识别→意图识别→4分类路由的完整对话流程
- **RAG 检索**：售后政策、物流说明、商品信息 3 个独立知识库，两段式检索（TopK=3）
- **多轮对话**：会话变量记录上下文，支持指代性问题理解
- **人工兜底**：愤怒用户自动转人工，提升客服效率
- **云端部署**：阿里云轻量应用服务器，公网可访问演示

### 演示地址

> 公网演示：http://39.96.31.124/chat/zBZzkKmKz3AtIj1q5

### 项目结构

```
02_dify_customer_service/
├── workflow_export.yaml   # Dify 工作流 DSL（结构示例，真实版本需从 Dify 平台导出）
└── README.md              # 详细说明
```

---

## 环境要求

- Python 3.10+
- 内存：8GB+（运行 BGE 模型）
- 可选：Docker（运行 MinIO / Milvus）

## 作者

胡蔚 - AI 应用开发工程师
- GitHub: [github.com/Fog-bound](https://github.com/Fog-bound)
