"""config 模块测试：.env 加载、优先级、refresh、脱敏描述。"""

import os

import pytest

from agent import config

_ALL_KEYS = [
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_TIMEOUT",
    "DEEPSEEK_MAX_RETRIES",
]


def _protect_config(monkeypatch):
    """注册全部 config 常量的自动还原（config.refresh 会就地改写它们）。"""
    for key in _ALL_KEYS:
        monkeypatch.setattr(config, key, getattr(config, key))


# ---------------------------------------------------------------------------
# .env 加载
# ---------------------------------------------------------------------------

def test_load_dotenv_parses_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=sk-from-dotenv\n"
        'DEEPSEEK_MODEL="deepseek-chat"\n'
        "# 注释行应被忽略\n"
        "没有等号的行应被忽略\n",
        encoding="utf-8",
    )
    try:
        config._load_dotenv(str(env_file))
        assert os.environ["DEEPSEEK_API_KEY"] == "sk-from-dotenv"
        assert os.environ["DEEPSEEK_MODEL"] == "deepseek-chat"  # 引号被剥离
    finally:
        os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ.pop("DEEPSEEK_MODEL", None)


def test_dotenv_does_not_override_real_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-real")
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=sk-from-dotenv\n", encoding="utf-8")
    config._load_dotenv(str(env_file))
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-real"


def test_load_dotenv_missing_file_is_noop(tmp_path):
    config._load_dotenv(str(tmp_path / "no-such.env"))  # 不应抛错


# ---------------------------------------------------------------------------
# refresh 与默认值
# ---------------------------------------------------------------------------

def test_refresh_reads_env_vars(monkeypatch):
    _protect_config(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    monkeypatch.setenv("DEEPSEEK_TIMEOUT", "12.5")
    monkeypatch.setenv("DEEPSEEK_MAX_RETRIES", "3")
    config.refresh()
    assert config.DEEPSEEK_API_KEY == "sk-test"
    assert config.DEEPSEEK_MODEL == "deepseek-reasoner"
    assert config.DEEPSEEK_TIMEOUT == 12.5
    assert config.DEEPSEEK_MAX_RETRIES == 3


def test_refresh_defaults_without_env(monkeypatch):
    _protect_config(monkeypatch)
    for key in _ALL_KEYS:
        monkeypatch.delenv(key, raising=False)
    config.refresh()
    assert config.DEEPSEEK_API_KEY == ""
    assert config.DEEPSEEK_MODEL == "deepseek-chat"
    assert config.DEEPSEEK_BASE_URL == "https://api.deepseek.com"
    assert config.DEEPSEEK_TIMEOUT == 60.0
    assert config.DEEPSEEK_MAX_RETRIES == 2


def test_refresh_rejects_bad_number(monkeypatch):
    _protect_config(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_TIMEOUT", "abc")
    with pytest.raises(ValueError, match="DEEPSEEK_TIMEOUT"):
        config.refresh()


# ---------------------------------------------------------------------------
# describe 脱敏
# ---------------------------------------------------------------------------

def test_describe_masks_api_key(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "sk-1234567890abcdef")
    text = config.describe()
    assert "sk-1234567890abcdef" not in text
    assert "deepseek-chat" in text
    assert "api_key=" in text


def test_describe_handles_missing_key(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")
    assert "(未设置)" in config.describe()
