"""Gradio app entrypoint for patent mining conversational agent."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from agent import AgentState, PatentAgent, build_dimension_summary

load_dotenv()
load_dotenv(".env.example")


def _init_agent() -> PatentAgent:
    """Initialize agent from environment configuration."""
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    max_rounds = int(os.getenv("MAX_ROUNDS", "12"))
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_deepseek_api_key":
        raise RuntimeError(
            "未检测到可用 API Key。请在 .env 中设置 OPENAI_API_KEY（或 DEEPSEEK_API_KEY）。"
        )

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return PatentAgent(
        model=model, max_rounds=max_rounds, temperature=temperature, client=client
    )


AGENT = _init_agent()


def _new_session_state() -> AgentState:
    """Create a clean state for each new UI session."""
    return AGENT.create_state()


def _render_status(state: AgentState) -> str:
    """Render current round and covered dimensions overview."""
    return f"对话轮数：{state.current_round}/{state.max_rounds} | 已覆盖维度：{state.covered_count}/10"


def _chat_handler(
    user_message: str,
    history: List[Dict[str, str]],
    state: AgentState,
) -> Tuple[str, AgentState, str, str, str]:
    """Main chat callback for each user message."""
    try:
        reply = AGENT.handle_user_message(state, user_message)
        status_text = _render_status(reply.state)
        dimensions_md = build_dimension_summary(reply.state)
        report_md = _extract_clean_report(reply.report_markdown or reply.message)
        chat_message = (
            "已完成专利交底书生成，请查看下方【最终报告】区域。"
            if report_md
            else reply.message
        )
        return chat_message, reply.state, status_text, dimensions_md, report_md
    except Exception as exc:
        # Keep the UI alive and show readable error instead of traceback crash.
        return f"系统异常：{exc}", state, _render_status(state), build_dimension_summary(state), ""


def _extract_clean_report(text: str) -> str:
    """Extract pure markdown report and remove JSON wrapper text."""
    if not text:
        return ""

    parsed = PatentAgent._try_parse_report_json(text)
    if parsed is not None and parsed.get("action") == "generate_report":
        extracted = parsed.get("report", "")
        raw = extracted if isinstance(extracted, str) else str(extracted)
        report = PatentAgent._normalize_report_markdown(raw)
        return _beautify_report(report)

    report = PatentAgent._normalize_report_markdown(text)
    if PatentAgent._looks_like_report_markdown(report):
        return _beautify_report(report)

    # Last fallback: strip accidental wrapper prefix/suffix from plain text.
    stripped = re.sub(
        r'^\s*\{\s*"action"\s*:\s*"generate_report"\s*,\s*"report"\s*:\s*',
        "",
        report,
    )
    stripped = re.sub(r'\s*\}\s*$', "", stripped)
    stripped = stripped.strip().strip('"')
    return _beautify_report(stripped) if PatentAgent._looks_like_report_markdown(stripped) else ""


def _beautify_report(report: str) -> str:
    """Normalize spacing for better markdown rendering."""
    text = report.replace("\r\n", "\n").strip()
    # Keep markdown readable with blank lines between major sections.
    text = re.sub(r"\n(##\s)", r"\n\n\1", text)
    text = re.sub(r"\n(###\s)", r"\n\n\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return _render_report_card(text)


def _render_report_card(report: str) -> str:
    """Render report in a boxed markdown card style."""
    lines = report.split("\n")
    quoted_lines: List[str] = []
    for line in lines:
        quoted_lines.append(f"> {line}" if line.strip() else ">")
    body = "\n".join(quoted_lines)
    return (
        "> <div align=\"center\" style=\"font-size: 34px; font-weight: 700; line-height: 1.4;\">专利交底书</div>\n"
        ">\n"
        "> ---\n"
        f"{body}\n"
        "> ---"
    )


with gr.Blocks(title="AI 专利挖掘助手") as demo:
    gr.Markdown("# AI 专利挖掘助手")

    session_state = gr.State(_new_session_state())

    with gr.Row():
        status_box = gr.Textbox(
            label="状态栏",
            value="对话轮数：0/12 | 已覆盖维度：0/10",
            interactive=False,
        )

    with gr.Row():
        dimensions_box = gr.Markdown(
            value="- ⬜ **技术实现**: 待补充\n- ⬜ **创新点**: 待补充\n- ⬜ **应用场景**: 待补充\n"
            "- ⬜ **用户群体**: 待补充\n- ⬜ **性能参数**: 待补充\n- ⬜ **材料选择**: 待补充\n"
            "- ⬜ **成本控制**: 待补充\n- ⬜ **现有技术差异**: 待补充\n- ⬜ **潜在扩展功能**: 待补充\n- ⬜ **使用限制**: 待补充"
        )
    report_box = gr.Markdown(label="最终报告", value="")

    gr.ChatInterface(
        fn=_chat_handler,
        additional_inputs=[session_state],
        additional_outputs=[session_state, status_box, dimensions_box, report_box],
        title=None,
        description="请输入你的发明想法，系统会在最多 12 轮内完成专利挖掘并生成交底书。",
    )


if __name__ == "__main__":
    demo.queue().launch()
