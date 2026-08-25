
from dataclasses import dataclass

from langchain_core.messages import AIMessage, ToolMessage

from .models import Brief
from .validate import Violation

# gpt-5.6-terra, intro pricing as of Aug 2026. Both agents use the same model
# on purpose — the architecture is the variable under test, not the model —
# so one pricing table covers both.
INPUT_PRICE_PER_MTOK = 2.0
OUTPUT_PRICE_PER_MTOK = 12.0


class RunFailed(Exception):
    def __init__(self, violations: list[Violation]):
        self.violations = violations
        super().__init__(f"brief failed validation after repair attempts: {violations}")


@dataclass
class RunTrace:
    """A Brief plus the trajectory data behind it — how many repair attempts
    it took, how many model/tool calls were made, and what it cost. Phase 3's
    eval harness needs this; the CLI just reads .brief off it.
    """

    brief: Brief
    repair_attempts: int  # 0 = passed validation on the first attempt
    initial_violations: int  # violation count on the *first* attempt, pre-repair
    model_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    latency_seconds: float

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self.input_tokens * INPUT_PRICE_PER_MTOK
            + self.output_tokens * OUTPUT_PRICE_PER_MTOK
        ) / 1_000_000


def new_message_stats(messages: list) -> tuple[int, int, int, int]:
    """Count model/tool calls and tokens in a slice of messages — must be
    called with only the messages a single agent.invoke() call produced, not
    the accumulated history, or repair attempts double-count the prior turn.
    """
    model_calls = tool_calls = input_tokens = output_tokens = 0
    for message in messages:
        if isinstance(message, AIMessage):
            model_calls += 1
            usage = message.usage_metadata or {}
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)
        elif isinstance(message, ToolMessage):
            tool_calls += 1
    return model_calls, tool_calls, input_tokens, output_tokens
