"""集中管理全部环境变量配置。

所有环境变量的读取统一收敛到本模块，其他模块只从 config 取值，
避免 os.environ 调用散落在各处。

配置优先级：真实环境变量 > 项目根目录 .env 文件 > 代码默认值
（.env 在模块首次导入时加载；已存在的环境变量不会被 .env 覆盖）。

用法：
    方式一（环境变量）：  export DEEPSEEK_API_KEY=sk-xxx
    方式二（.env 文件）：  cp .env.example .env 后编辑填入
"""

import os

# 项目根目录（agent/ 的上一级），.env 放在这里
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv(path=None):
    """极简 .env 加载器（无第三方依赖）：逐行解析 KEY=VALUE，支持 # 注释与引号剥离。"""
    path = path or os.path.join(_PROJECT_ROOT, ".env")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:  # 真实环境变量优先
                os.environ[key] = value


def _env_str(name, default):
    return os.environ.get(name, default)


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"环境变量 {name} 应为数字，当前值: {raw!r}") from None


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"环境变量 {name} 应为整数，当前值: {raw!r}") from None


def refresh():
    """（重新）从环境变量读取全部配置到模块级常量。模块导入时自动执行一次。"""
    global DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL
    global DEEPSEEK_TIMEOUT, DEEPSEEK_MAX_RETRIES
    DEEPSEEK_API_KEY = _env_str("DEEPSEEK_API_KEY", "")
    DEEPSEEK_MODEL = _env_str("DEEPSEEK_MODEL", "deepseek-chat")
    DEEPSEEK_BASE_URL = _env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_TIMEOUT = _env_float("DEEPSEEK_TIMEOUT", 60.0)
    DEEPSEEK_MAX_RETRIES = _env_int("DEEPSEEK_MAX_RETRIES", 2)


def describe():
    """返回脱敏的配置概览（启动日志展示用，避免泄露完整 API Key）。"""
    key = DEEPSEEK_API_KEY
    if not key:
        masked = "(未设置)"
    elif len(key) > 10:
        masked = f"{key[:6]}...{key[-4:]}"
    else:
        masked = "(已设置)"
    return (
        f"model={DEEPSEEK_MODEL}, base_url={DEEPSEEK_BASE_URL}, "
        f"timeout={DEEPSEEK_TIMEOUT}s, max_retries={DEEPSEEK_MAX_RETRIES}, "
        f"api_key={masked}"
    )


_load_dotenv()
refresh()
