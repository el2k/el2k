"""LLM 层测试：文本协议解析、FakeLLM、DeepSeekLLM（stub 掉网络客户端）。"""

from types import SimpleNamespace

import pytest

from agent import config
from agent.exceptions import LLMError, ParseError
from agent.llm import (
    DeepSeekLLM,
    FakeLLM,
    parse_text_response,
    text_response,
    tool_call_response,
)


# ---------------------------------------------------------------------------
# parse_text_response：LLM 输出解析逻辑
# ---------------------------------------------------------------------------

def test_parse_plain_text_as_answer():
    result = parse_text_response("北京是中国的首都。")
    assert result == {"thought": "", "tool_calls": [], "answer": "北京是中国的首都。"}


def test_parse_empty():
    assert parse_text_response("")["answer"] == ""
    assert parse_text_response("   \n")["tool_calls"] == []


def test_parse_json_protocol():
    text = '{"thought": "查一下天气", "tool_calls": [{"name": "weather", "args": {"city": "北京"}}], "answer": ""}'
    result = parse_text_response(text)
    assert result["thought"] == "查一下天气"
    assert result["tool_calls"][0]["name"] == "weather"
    assert result["tool_calls"][0]["args"] == {"city": "北京"}
    assert result["answer"] == ""


def test_parse_xml_protocol_single_invoke():
    text = '<thought>需要计算</thought><invoke name="calculator">{"expression": "2+3"}</invoke><answer></answer>'
    result = parse_text_response(text)
    assert result["thought"] == "需要计算"
    assert result["tool_calls"][0]["name"] == "calculator"
    assert result["tool_calls"][0]["args"] == {"expression": "2+3"}


def test_parse_xml_protocol_multiple_invokes():
    text = (
        '<thought>同时查两个城市</thought>'
        '<invoke name="weather">{"city": "北京"}</invoke>'
        '<invoke name="weather">{"city": "上海"}</invoke>'
        '<answer></answer>'
    )
    result = parse_text_response(text)
    assert len(result["tool_calls"]) == 2
    assert result["tool_calls"][1]["args"] == {"city": "上海"}
    # 多个调用应有不同的 id，保证与 tool 消息配对
    ids = [tc["id"] for tc in result["tool_calls"]]
    assert len(set(ids)) == 2


def test_parse_xml_with_params_wrapper_and_fences():
    text = '<invoke name="search"><params>```json\n{"query": "agent"}\n```</params></invoke>'
    result = parse_text_response(text)
    assert result["tool_calls"][0]["args"] == {"query": "agent"}


def test_parse_non_json_invoke_args_fallback():
    text = '<invoke name="search">关键词 agent 框架</invoke>'
    result = parse_text_response(text)
    assert result["tool_calls"][0]["args"] == {"query": "关键词 agent 框架"}


def test_parse_thought_and_answer_without_tools():
    text = "<thought>简单问题</thought><answer>答案是 42</answer>"
    result = parse_text_response(text)
    assert result["thought"] == "简单问题"
    assert result["answer"] == "答案是 42"
    assert result["tool_calls"] == []


def test_parse_unrelated_json_treated_as_answer():
    result = parse_text_response('{"foo": "bar"}')
    # 不含 thought/tool_calls/answer 键的 JSON 不视为协议，整体作为答案
    assert result["answer"] == '{"foo": "bar"}'


# ---------------------------------------------------------------------------
# FakeLLM
# ---------------------------------------------------------------------------

def test_fake_llm_returns_queued_responses_in_order():
    llm = FakeLLM([text_response("第一"), text_response("第二")])
    assert llm.chat([{"role": "user", "content": "hi"}])["answer"] == "第一"
    assert llm.chat([])["answer"] == "第二"


def test_fake_llm_records_requests():
    llm = FakeLLM([text_response("ok")])
    llm.chat([{"role": "user", "content": "问题"}], tools=[{"type": "function"}], session_id="s1")
    assert llm.received[0]["messages"][0]["content"] == "问题"
    assert llm.received[0]["tools"] == [{"type": "function"}]
    assert llm.received[0]["session_id"] == "s1"


def test_fake_llm_exhausted_returns_placeholder():
    llm = FakeLLM()
    assert "预设响应" in llm.chat([])["answer"]


def test_response_helpers():
    assert text_response("hi")["tool_calls"] == []
    tc = tool_call_response("calc", {"expression": "1+1"}, call_id="fixed-id")
    assert tc["tool_calls"][0]["id"] == "fixed-id"
    assert tc["tool_calls"][0]["name"] == "calc"


# ---------------------------------------------------------------------------
# DeepSeekLLM（不联网，stub 客户端）
# ---------------------------------------------------------------------------

def _api_response(content="", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _api_tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _make_llm_with_stub(outcomes, max_retries=2):
    llm = DeepSeekLLM(api_key="test-key", max_retries=max_retries)

    class StubCompletions:
        calls = 0

        def create(self, **kwargs):
            StubCompletions.calls += 1
            item = outcomes.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=StubCompletions()))
    return llm


_CONFIG_KEYS = [
    "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "DEEPSEEK_BASE_URL",
    "DEEPSEEK_TIMEOUT", "DEEPSEEK_MAX_RETRIES",
]


def _protect_config(monkeypatch):
    """注册全部 config 常量的自动还原（config.refresh 会就地改写它们）。"""
    for key in _CONFIG_KEYS:
        monkeypatch.setattr(config, key, getattr(config, key))


def test_deepseek_requires_api_key(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")
    with pytest.raises(LLMError, match="DEEPSEEK_API_KEY"):
        DeepSeekLLM()


def test_deepseek_reads_config_from_env(monkeypatch):
    """环境变量（或 .env）中的配置经 config 模块生效。"""
    _protect_config(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    config.refresh()
    llm = DeepSeekLLM()
    assert llm.api_key == "sk-env"
    assert llm.model == "deepseek-reasoner"
    assert llm.base_url == "https://api.deepseek.com"  # 默认值


def test_deepseek_explicit_params_override_config():
    """显式构造参数优先于 config 配置。"""
    llm = DeepSeekLLM(api_key="k", model="m", base_url="b", timeout=5, max_retries=0)
    assert llm.api_key == "k"
    assert llm.model == "m"
    assert llm.base_url == "b"
    assert llm.timeout == 5
    assert llm.max_retries == 0


def test_deepseek_parses_native_tool_calls():
    llm = _make_llm_with_stub([
        _api_response(
            content="让我算一下",
            tool_calls=[_api_tool_call("call_1", "calculator", '{"expression": "2+3"}')],
        )
    ])
    result = llm.chat([{"role": "user", "content": "算 2+3"}],
                      tools=[{"type": "function", "function": {"name": "calculator"}}])
    assert result["thought"] == "让我算一下"
    assert result["tool_calls"] == [{"id": "call_1", "name": "calculator", "args": {"expression": "2+3"}}]
    assert result["answer"] == ""


def test_deepseek_parses_plain_content_via_text_protocol():
    llm = _make_llm_with_stub([
        _api_response(content="<thought>不需要工具</thought><answer>直接回答</answer>")
    ])
    result = llm.chat([{"role": "user", "content": "你好"}])
    assert result["answer"] == "直接回答"
    assert result["thought"] == "不需要工具"


def test_deepseek_bad_tool_arguments_raises_parse_error():
    llm = _make_llm_with_stub([
        _api_response(tool_calls=[_api_tool_call("c1", "calc", "不是json")])
    ])
    with pytest.raises(ParseError):
        llm.chat([{"role": "user", "content": "x"}])


def test_deepseek_retries_on_transient_error(monkeypatch):
    monkeypatch.setattr("agent.llm.time.sleep", lambda s: None)
    llm = _make_llm_with_stub([
        RuntimeError("connection reset"),       # 第 1 次失败
        RuntimeError("timeout"),                # 第 2 次失败
        _api_response(content="重试成功"),      # 第 3 次成功
    ], max_retries=2)
    result = llm.chat([{"role": "user", "content": "hi"}])
    assert result["answer"] == "重试成功"


def test_deepseek_raises_llm_error_after_retries_exhausted(monkeypatch):
    monkeypatch.setattr("agent.llm.time.sleep", lambda s: None)
    llm = _make_llm_with_stub([RuntimeError("down")] * 3, max_retries=2)
    with pytest.raises(LLMError, match="已重试 2 次"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_deepseek_does_not_retry_parse_errors():
    llm = _make_llm_with_stub([
        _api_response(tool_calls=[_api_tool_call("c1", "calc", "bad-json")]),
        _api_response(content="不应被调用"),
    ])
    with pytest.raises(ParseError):
        llm.chat([{"role": "user", "content": "x"}])
