"""Shared text utilities."""
from __future__ import annotations

import re

_ws = re.compile(r"\s+")


def norm(s: str) -> str:
    return _ws.sub(" ", (s or "").lower()).strip()


def clauses(text: str) -> list[str]:
    """Split a narrative into short clauses for term matching."""
    parts = re.split(r"[.;:\n,()]|\band\b|\bwith\b|\bthen\b", text or "", flags=re.I)
    return [p.strip() for p in parts if len(p.strip()) >= 3]
