"""全局配置：统一从 .env 读取，避免 os.getenv 散落在各模块。

设计取舍：
- 所有字段都有默认值，密钥类字段做**延迟校验**（用到时才报错）。
  这样 `--rebuild` 这种只需要 Embedding 的场景，不会被缺失的对话模型 Key 卡住。
- 路径全部基于 __file__ 推导，不受运行时工作目录影响。
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ===== Embedding（走 OpenAI 兼容接口，DeepSeek 不提供向量模型）=====
    openai_api_key: str = ""
    openai_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"

    # ===== 对话模型 =====
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    llm_temperature: float = 0.2

    # ===== Agent 行为 =====
    # 单轮问答允许的最大"模型决策轮次"，超过后强制收敛，防止死循环烧钱
    agent_max_steps: int = 8
    # 单次检索返回的片段数。必须与切块粒度同调：chunk_size 从 800 降到 350 后，
    # 一个块只覆盖半页，k=4 最多只能覆盖 2 页；实测检索命中 k=4 时 14/23，
    # k=8 时 21/23。块越细，k 要越大，否则是负优化。
    retrieval_top_k: int = 8

    # ===== 路径 =====
    data_dir: Path = PROJECT_ROOT / "data"
    index_dir: Path = PROJECT_ROOT / "faiss_index"
    report_dir: Path = PROJECT_ROOT / "reports"

    def require_deepseek(self) -> str:
        """对话模型 Key 缺失时才报错，便于索引类操作独立运行。"""
        if not self.deepseek_api_key:
            raise RuntimeError(
                "未检测到 DEEPSEEK_API_KEY，请检查项目根目录 .env 是否已保存"
            )
        return self.deepseek_api_key

    def require_embedding(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError(
                "未检测到 OPENAI_API_KEY，请保存 .env 后重试（注意别只在编辑器里改而未落盘）"
            )
        return self.openai_api_key


settings = Settings()
