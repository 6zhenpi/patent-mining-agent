"""Terminal demo entrypoint for patent mining agent."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from agent import PatentAgent, build_dimension_summary


def _init_agent() -> PatentAgent:
    """Initialize agent from environment variables."""
    load_dotenv()
    load_dotenv(".env.example")

    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    max_rounds = int(os.getenv("MAX_ROUNDS", "12"))
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_deepseek_api_key":
        raise RuntimeError("请先在 .env 设置 OPENAI_API_KEY（或 DEEPSEEK_API_KEY）。")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return PatentAgent(
        model=model,
        max_rounds=max_rounds,
        temperature=temperature,
        client=client,
    )


def main() -> None:
    """Run a terminal interactive demo."""
    agent = _init_agent()
    state = agent.create_state()

    print("=== AI 专利挖掘助手（CLI 演示版）===")
    print("输入 exit 结束。建议首句：我想发明一个能自动分类收纳衣物的衣柜。")

    while True:
        user_input = input("\n你：").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("已退出。")
            break

        try:
            reply = agent.handle_user_message(state, user_input)
        except Exception as exc:
            print(f"\n助手：系统异常：{exc}")
            continue

        state = reply.state
        print(f"\n助手：{reply.message}")
        print(
            f"\n[状态] 对话轮数：{state.current_round}/{state.max_rounds} | "
            f"已覆盖维度：{state.covered_count}/10"
        )
        print("\n[维度摘要]")
        print(build_dimension_summary(state))

        if reply.report_markdown:
            print("\n[最终交底书]")
            print(reply.report_markdown)
            print("\n对话已结束。")
            break


if __name__ == "__main__":
    main()
