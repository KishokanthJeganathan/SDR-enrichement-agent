"""Pydantic schemas for the SDR account brief. See CLAUDE.md §3.

These models enforce structure only: field presence and types. Cross-field
and cross-object rules (an evidence index must exist in `sources`, an
inference must rest on verified evidence) live in validate.py, since they
need the whole Brief to check and don't map onto a single model's fields.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["HIGH", "MEDIUM", "LOW"]
ClaimKind = Literal["verified", "inference"]


class Enquiry(BaseModel):
    contact_name: str  # passed through, never researched
    contact_email: str  # domain is used; the address is not researched
    company_name: str
    message: str


class Source(BaseModel):
    index: int
    url: str
    fetched_at: datetime
    published_at: datetime | None = None


class Claim(BaseModel):
    value: str
    confidence: Confidence
    evidence: list[int]
    kind: ClaimKind


class Conflict(BaseModel):
    field: str
    positions: list[str]
    evidence: list[int]
    resolution: str


class Brief(BaseModel):
    company: Claim
    funding: Claim | None = None
    hiring_signals: list[Claim] = Field(default_factory=list)
    likely_use_case: Claim
    deal_band: Claim | None = None
    unverified: list[str] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    suggested_opening: str
    sources: list[Source]
    overall_confidence: Confidence
    generated_at: datetime
