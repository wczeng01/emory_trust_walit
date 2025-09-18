# langserve_trust/trust_adapter/schemas.py
from __future__ import annotations
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: str
    content: str


class ShimQuery(BaseModel):
    # Minimal fields used by Conversation.conv_question(...)
    question: str = ""


class ShimVariable(BaseModel):
    # Minimal fields used by Conversation.conv_* methods
    vid: str = ""
    metadata: str = ""
    # The Conversation code only reads `queries` for transition and expects each to have .question
    queries: List[ShimQuery] = Field(default_factory=list)


class TrustInput(BaseModel):
    # Which Conversation API do you want to call?
    #   "da_tags" | "conv_transition" | "conv_question"
    op: str = Field("da_tags", description="Operation to run on Conversation")

    # Common
    session_id: str = ""
    model_name: str = "gpt-4o"                          # can override at runtime
    template_file: str = ""                             # '' => use server default
    simulate_user: str = ""                             # '' => disabled
    history: List[ChatTurn] = Field(default_factory=list)

    # For da_tags
    questions: List[str] = Field(default_factory=list)

    # For conv_transition / conv_question
    variable: ShimVariable = Field(default_factory=ShimVariable)

    # For conv_question
    cur_query: ShimQuery = Field(default_factory=ShimQuery)
    cur_tags: List[str] = Field(default_factory=list)

    # Extra per-request knobs (unused by Conversation itself but future-proof)
    config: Dict[str, Any] = Field(default_factory=dict)


class TrustOutput(BaseModel):
    text: str = ""
    state: Dict[str, Any] = Field(default_factory=dict)
