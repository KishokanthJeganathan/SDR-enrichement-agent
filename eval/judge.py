"""Citation-validity judge. See CLAUDE.md §7 — judge discipline: binary
verdict, one criterion per call, reasoning before verdict, temperature 0.

Deliberately NOT gpt-5.6-* (the research agent's model family): those are
reasoning-tier models that force temperature=1 with no override, which
would violate the temperature-0 requirement outright. gpt-4o-mini is a
classic (non-reasoning) model that still honors temperature, and is cheap
enough that judging every claim in a brief costs very little.
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()  # ChatOpenAI reads OPENAI_API_KEY at construction, below

JUDGE_MODEL = "gpt-4o-mini"

_EVIDENCE_CHARS = 4000  # keep judge calls small; sources are already capped at fetch time


class _Verdict(BaseModel):
    reasoning: str = Field(description="Brief reasoning, written before the verdict.")
    supported: bool = Field(description="True only if the source text directly supports the claim.")


_judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(_Verdict)

_JUDGE_PROMPT = """You are checking whether a single factual claim is supported \
by a piece of source text.

Claim: {claim}

Source text:
{evidence}

Does the source text support the claim? The source must actually state or clearly \
imply the claim — being merely topically related is not enough. Write your \
reasoning first, then give your verdict."""


def judge_citation(claim: str, evidence_text: str) -> tuple[bool, str]:
    """Returns (supported, reasoning)."""
    verdict = _judge_llm.invoke(
        _JUDGE_PROMPT.format(claim=claim, evidence=evidence_text[:_EVIDENCE_CHARS])
    )
    return verdict.supported, verdict.reasoning


def validate_judge(labels_path: Path) -> tuple[float, list[dict]]:
    """Run the judge against hand-labeled (claim, evidence, expected) examples.
    Returns (accuracy, mismatches) so a caller can decide whether to trust
    the judge's numbers on the actual golden-set run.
    """
    labels = json.loads(labels_path.read_text())
    mismatches = []
    correct = 0
    for example in labels:
        predicted, reasoning = judge_citation(example["claim"], example["evidence"])
        if predicted == example["expected"]:
            correct += 1
        else:
            mismatches.append({**example, "predicted": predicted, "judge_reasoning": reasoning})
    return correct / len(labels), mismatches
