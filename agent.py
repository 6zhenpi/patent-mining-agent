"""Core patent conversation agent with dynamic state tracking."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from openai import APIStatusError, OpenAI
from pydantic import BaseModel, Field

from prompts import DIMENSIONS, FOLLOW_UP_PROMPT, STATE_UPDATE_PROMPT, SYSTEM_PROMPT

DimensionStatus = Literal["missing", "partial", "covered"]


class DimensionItem(BaseModel):
    """Single mining dimension state."""

    status: DimensionStatus = "missing"
    content: str = ""


class AgentState(BaseModel):
    """Whole conversation state for one user session."""

    dimensions: Dict[str, DimensionItem] = Field(
        default_factory=lambda: {name: DimensionItem() for name in DIMENSIONS}
    )
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    current_round: int = 0
    max_rounds: int = 12
    finished: bool = False

    @property
    def covered_count(self) -> int:
        """Count dimensions that are not missing."""
        return sum(1 for item in self.dimensions.values() if item.status != "missing")


@dataclass
class AgentReply:
    """Structured output for UI rendering."""

    message: str
    report_markdown: Optional[str]
    state: AgentState


class PatentAgent:
    """Stateful patent-mining dialogue agent backed by OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        max_rounds: int = 12,
        temperature: float = 0.3,
        client: Optional[OpenAI] = None,
    ) -> None:
        self.client = client or OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_rounds = max_rounds

    def create_state(self) -> AgentState:
        """Create initial agent state for new chat."""
        return AgentState(max_rounds=self.max_rounds)

    def handle_user_message(self, state: AgentState, user_message: str) -> AgentReply:
        """Handle one user turn and return assistant message/report."""
        if state.finished:
            return AgentReply(
                message="本轮对话已结束，如需新专利挖掘，请刷新页面重新开始。",
                report_markdown=None,
                state=state,
            )

        state.current_round += 1
        state.conversation_history.append({"role": "user", "content": user_message})

        updates, unknown_detected, unknown_dimensions = self._analyze_and_update_state(
            state, user_message
        )
        self._merge_dimension_updates(state, updates)

        should_finish = self._should_finish(state)
        assistant_text = self._generate_next_response(
            state=state,
            unknown_detected=unknown_detected,
            unknown_dimensions=unknown_dimensions,
            should_finish=should_finish,
        )

        report: Optional[str] = None
        parsed = self._try_parse_report_json(assistant_text)
        if parsed is not None and parsed.get("action") == "generate_report":
            raw_report = parsed.get("report", "")
            report = self._normalize_report_markdown(
                raw_report if isinstance(raw_report, str) else str(raw_report)
            )
            assistant_text = report or "已完成报告生成。"
            state.finished = True
        elif should_finish:
            # Fallback: if model did not output strict JSON, force report generation.
            report = self._normalize_report_markdown(self._generate_report_directly(state))
            assistant_text = report
            state.finished = True
        elif self._looks_like_report_markdown(assistant_text):
            # Some providers may skip wrapper JSON and output report directly.
            report = self._normalize_report_markdown(assistant_text)
            state.finished = True

        if state.finished and not report:
            report = self._normalize_report_markdown(assistant_text)
            assistant_text = report or "已完成报告生成。"

        state.conversation_history.append({"role": "assistant", "content": assistant_text})

        return AgentReply(message=assistant_text, report_markdown=report, state=state)

    def _analyze_and_update_state(
        self, state: AgentState, user_message: str
    ) -> Tuple[Dict[str, Dict[str, str]], bool, List[str]]:
        """Use LLM to extract multi-dimension updates from latest user reply."""
        dims_payload = {
            name: state.dimensions[name].model_dump() for name in state.dimensions.keys()
        }
        payload = {
            "dimensions": dims_payload,
            "latest_user_message": user_message,
            "history_tail": state.conversation_history[-6:],
        }

        response = self._chat_complete(
            messages=[
                {"role": "system", "content": STATE_UPDATE_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed: Dict[str, Any] = self._safe_json_load(content)

        updates = parsed.get("updates", {}) if isinstance(parsed, dict) else {}
        unknown_detected = bool(parsed.get("unknown_detected", False)) if isinstance(parsed, dict) else False
        unknown_dimensions = parsed.get("unknown_dimensions", []) if isinstance(parsed, dict) else []
        if not isinstance(unknown_dimensions, list):
            unknown_dimensions = []

        # Deterministic fallback: explicitly capture common "unknown" replies.
        if self._contains_unknown_intent(user_message):
            unknown_detected = True
            if not unknown_dimensions:
                unknown_dimensions = [
                    name for name, item in state.dimensions.items() if item.status == "missing"
                ][:2]

        return updates, unknown_detected, [str(d) for d in unknown_dimensions]

    def _merge_dimension_updates(
        self, state: AgentState, updates: Dict[str, Dict[str, str]]
    ) -> None:
        """Merge LLM extracted updates into in-memory state."""
        for name, item in updates.items():
            if name not in state.dimensions:
                continue
            status = item.get("status", state.dimensions[name].status)
            content = item.get("content", "").strip()

            if status not in {"missing", "partial", "covered"}:
                status = state.dimensions[name].status

            # Keep richer state when previous is covered.
            previous = state.dimensions[name]
            if previous.status == "covered" and status in {"missing", "partial"}:
                status = "covered"

            merged_content = previous.content
            if content and content not in merged_content:
                merged_content = f"{merged_content}；{content}".strip("；") if merged_content else content

            state.dimensions[name] = DimensionItem(status=status, content=merged_content)

    def _should_finish(self, state: AgentState) -> bool:
        """Stop if all dimensions are covered or round limit reached."""
        all_covered = all(item.status == "covered" for item in state.dimensions.values())
        round_limit = state.current_round >= state.max_rounds
        return all_covered or round_limit

    def _generate_next_response(
        self,
        state: AgentState,
        unknown_detected: bool,
        unknown_dimensions: List[str],
        should_finish: bool,
    ) -> str:
        """Generate either follow-up question or final report JSON."""
        dims_payload = {
            name: state.dimensions[name].model_dump() for name in state.dimensions.keys()
        }
        control_payload = {
            "current_round": state.current_round,
            "max_rounds": state.max_rounds,
            "covered_count": state.covered_count,
            "dimensions": dims_payload,
            "unknown_detected": unknown_detected,
            "unknown_dimensions": unknown_dimensions,
            "should_finish": should_finish,
            "history": state.conversation_history,
        }

        response = self._chat_complete(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": FOLLOW_UP_PROMPT},
                {"role": "user", "content": json.dumps(control_payload, ensure_ascii=False)},
            ],
            temperature=self.temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def _generate_report_directly(self, state: AgentState) -> str:
        """Fallback report generation when strict JSON is not returned."""
        dims_payload = {
            name: state.dimensions[name].model_dump() for name in state.dimensions.keys()
        }
        response = self._chat_complete(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "请立即输出最终 Markdown 交底书，不要提问。"
                        + json.dumps(
                            {
                                "current_round": state.current_round,
                                "max_rounds": state.max_rounds,
                                "dimensions": dims_payload,
                                "history": state.conversation_history,
                            },
                            ensure_ascii=False,
                        )
                    ),
                },
            ],
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()

    def _chat_complete(self, messages: List[Dict[str, str]], temperature: float, **kwargs: Any):
        """Centralize model calls and convert provider errors to readable messages."""
        try:
            return self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=messages,
                **kwargs,
            )
        except APIStatusError as exc:
            err_msg = ""
            body = getattr(exc, "body", None)
            if isinstance(body, dict):
                error = body.get("error", {})
                if isinstance(error, dict):
                    err_msg = str(error.get("message", "")).strip()
            lowered = err_msg.lower()
            if exc.status_code == 402 or "insufficient balance" in lowered:
                raise RuntimeError(
                    "调用模型失败：DeepSeek 账户余额不足（402 Insufficient Balance）。请先充值后重试。"
                ) from exc
            raise RuntimeError(
                f"调用模型失败（HTTP {exc.status_code}）：{err_msg or '请检查 API Key、Base URL 或模型名配置'}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"调用模型失败：{exc}") from exc

    @staticmethod
    def _try_parse_report_json(text: str) -> Optional[Dict[str, Any]]:
        """Try parse strict report JSON from assistant output."""
        parsed = PatentAgent._safe_json_load(text)
        if isinstance(parsed, dict) and "action" in parsed:
            return parsed
        fenced_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        if fenced_match:
            parsed = PatentAgent._safe_json_load(fenced_match.group(1))
            if isinstance(parsed, dict) and "action" in parsed:
                return parsed
        object_match = re.search(r"(\{[\s\S]*\"action\"[\s\S]*\})", text)
        if object_match:
            parsed = PatentAgent._safe_json_load(object_match.group(1))
            if isinstance(parsed, dict) and "action" in parsed:
                return parsed
        report_from_wrapper = PatentAgent._extract_report_from_wrapper(text)
        if report_from_wrapper is not None:
            return {"action": "generate_report", "report": report_from_wrapper}
        return None

    @staticmethod
    def _extract_report_from_wrapper(text: str) -> Optional[str]:
        """Best-effort extraction for malformed JSON wrapper outputs."""
        if "generate_report" not in text or "report" not in text:
            return None
        report_key_match = re.search(r'"report"\s*:\s*', text)
        if not report_key_match:
            return None
        value_part = text[report_key_match.end() :].strip()
        if not value_part:
            return None
        if value_part[0] == '"':
            escaped = []
            i = 1
            while i < len(value_part):
                ch = value_part[i]
                if ch == '"' and value_part[i - 1] != "\\":
                    break
                escaped.append(ch)
                i += 1
            candidate = "".join(escaped)
            try:
                return json.loads(f'"{candidate}"')
            except json.JSONDecodeError:
                return candidate.replace("\\n", "\n")
        stripped = value_part.rstrip("} \n\t")
        return stripped.strip()

    @staticmethod
    def _normalize_report_markdown(text: str) -> str:
        """Normalize escaped report text to proper markdown."""
        report = text.strip()
        if not report:
            return report
        if "\\n" in report:
            try:
                escaped = report.replace("\\", "\\\\").replace('"', '\\"')
                decoded = json.loads(f'"{escaped}"')
                if isinstance(decoded, str):
                    report = decoded
            except json.JSONDecodeError:
                report = report.replace("\\n", "\n")
        return report.strip()

    @staticmethod
    def _looks_like_report_markdown(text: str) -> bool:
        """Heuristic check for direct markdown report output."""
        stripped = text.strip()
        if not stripped:
            return False
        return stripped.startswith("# 发明名称") or "## 一、摘要" in stripped

    @staticmethod
    def _safe_json_load(text: str) -> Dict[str, Any]:
        """Safely parse JSON text to dict, return empty on failure."""
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _contains_unknown_intent(text: str) -> bool:
        """Detect if user explicitly says they don't know yet."""
        lowered = text.strip().lower()
        keywords = [
            "不知道",
            "没想好",
            "不清楚",
            "暂时不确定",
            "不太确定",
            "i don't know",
            "not sure",
        ]
        return any(k in lowered for k in keywords)


def build_dimension_summary(state: AgentState) -> str:
    """Render concise markdown summary for dimension coverage."""
    lines: List[str] = []
    for name in DIMENSIONS:
        item = state.dimensions[name]
        status_emoji = {"missing": "⬜", "partial": "🟨", "covered": "🟩"}[item.status]
        snippet = item.content[:80] + ("..." if len(item.content) > 80 else "") if item.content else "待补充"
        lines.append(f"- {status_emoji} **{name}**: {snippet}")
    return "\n".join(lines)
