"""通用小工具函数。"""

import json


def truncate(text, max_chars, suffix="...(已截断)"):
    """截断过长文本，用于控制 context 与日志体积。"""
    text = str(text)
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars] + suffix


def to_display_text(result):
    """把工具返回值（任意可序列化结构）转为可放入消息/日志的文本。"""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(result)
