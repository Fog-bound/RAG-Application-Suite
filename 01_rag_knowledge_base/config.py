# config.py —— 统一配置文件（密钥请通过环境变量或 .env 设置，不要写死在代码里）
import os
from dotenv import load_dotenv

# 读取项目根目录下的 .env 文件（.env 已被 .gitignore 排除，不会上传到 GitHub）
load_dotenv()

# ============ DeepSeek 大模型（生成回答用） ============
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL_NAME = os.environ.get("DEEPSEEK_MODEL_NAME", "deepseek-chat")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# ============ Rerank 重排模型（XInference 本地服务） ============
RERANK_API_KEY = os.environ.get("RERANK_API_KEY", "")
RERANK_URL = os.environ.get("RERANK_URL", "http://localhost:9997/v1/rerank")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "bge-reranker-v2-m3")

# ============ MinIO 对象存储 ============
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "your-minio-secret-key")
