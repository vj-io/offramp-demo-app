"""Domain types shared across HelpHub's support workflows."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class Queue(StrEnum):
    """The support queues an incoming ticket can be routed to."""

    BILLING = "billing"
    SHIPPING = "shipping"
    RETURNS_REFUNDS = "returns_refunds"
    TECHNICAL_SUPPORT = "technical_support"
    ACCOUNT_ACCESS = "account_access"
    GENERAL = "general"


class ThreadMessage(BaseModel):
    """One message in a ticket thread."""

    author: str
    role: Literal["customer", "agent", "internal_note", "system"]
    sent_at: str  # ISO 8601 as exported by the helpdesk; not parsed
    text: str


class ToolResult(BaseModel):
    """What a tool returned during an agent run."""

    tool: str
    output: str
