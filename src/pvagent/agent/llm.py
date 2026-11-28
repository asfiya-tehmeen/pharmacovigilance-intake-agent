"""OpenAI-compatible chat client for Groq / Gemini / Ollama with a disk cache and a daily token budget.

Every response is cached by (provider, model, messages, temperature, sample index), so reruns cost nothing.
Token usage is logged per call; once today's usage reaches the budget, BudgetExceeded is raised and the
caller can stop cleanly and resume tomorrow.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

from ..config import env, path

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY"),
    "ollama": (None, None),
}


def _redact(text: str) -> str:
    """Provider error messages include the account's organisation id; keep it out of saved results and logs."""
    return re.sub(r"org_[A-Za-z0-9]+", "org_<redacted>", text)


class BudgetExceeded(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    cached: bool
    model: str


class LLMClient:
    def __init__(self, provider: str, model: str, daily_token_budget: int = 190_000, max_tokens: int = 1200,
                 cache_path: Path | None = None, timeout: float = 90.0):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider}")
        self.provider, self.model = provider, model
        self.budget, self.max_tokens = daily_token_budget, max_tokens
        url, key_env = PROVIDERS[provider]
        if provider == "ollama":
            url = env("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/v1/chat/completions"
            key = "ollama"
        else:
            key = env(key_env)
            if not key:
                raise LLMError(f"{key_env} is not set. Copy .env.example to .env and add your key.")
        self.url, self.key = url, key
        self.key_id = hashlib.sha256(key.encode()).hexdigest()[:8]   # budget is tracked per key, never stored in full
        self.http = httpx.Client(timeout=timeout)
        cp = cache_path or path("data/cache/llm_cache.sqlite")
        cp.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(cp))
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS usage (day TEXT, provider TEXT, model TEXT, prompt INT, "
                        "completion INT, seconds REAL, ts REAL)")
        if "key_id" not in [r[1] for r in self.db.execute("PRAGMA table_info(usage)").fetchall()]:
            self.db.execute("ALTER TABLE usage ADD COLUMN key_id TEXT")   # older rows stay NULL = previous key
        self.db.commit()

    def used_today(self) -> int:
        r = self.db.execute("SELECT coalesce(sum(prompt+completion),0) FROM usage WHERE day=? AND provider=? AND model=? AND key_id IS ?",
                            (date.today().isoformat(), self.provider, self.model, self.key_id)).fetchone()
        return int(r[0])

    def _key(self, messages, temperature, sample) -> str:
        blob = json.dumps([self.provider, self.model, messages, temperature, sample, self.max_tokens], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages: list[dict], temperature: float = 0.0, sample: int = 0, json_mode: bool = True) -> LLMResponse:
        k = self._key(messages, temperature, sample)
        hit = self.db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        if hit:
            d = json.loads(hit[0])
            return LLMResponse(d["text"], d["pt"], d["ct"], d["s"], True, self.model)
        if self.used_today() >= self.budget:
            raise BudgetExceeded(f"{self.provider}/{self.model}: daily budget {self.budget} tokens reached")

        body: dict = {"model": self.model, "messages": messages, "temperature": temperature,
                      "max_tokens": self.max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.provider == "groq" and "gpt-oss" in self.model:
            body["reasoning_effort"] = "low"     # keeps hidden reasoning tokens (and cost) down
        headers = {"Authorization": f"Bearer {self.key}"}
        t0 = time.time()
        for attempt in range(6):
            try:
                r = self.http.post(self.url, json=body, headers=headers)
            except httpx.TransportError as e:       # DNS failure, connection reset, timeout: transient, retry
                if attempt == 5:
                    raise LLMError(f"network error after retries: {e}") from e
                time.sleep(min(60, 2 ** attempt * 3))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("retry-after", 0) or 0) or min(60, 2 ** attempt * 2)
                if "tokens per day" in r.text.lower() or "per day" in r.text.lower():
                    raise BudgetExceeded(f"provider daily limit: {_redact(r.text)[:200]}")
                time.sleep(wait)
                continue
            if r.status_code >= 400:
                raise LLMError(f"{r.status_code}: {_redact(r.text)[:500]}")
            break
        else:
            raise LLMError(f"gave up after retries: {r.status_code} {r.text[:200]}")
        secs = time.time() - t0
        d = r.json()
        text = d["choices"][0]["message"].get("content") or ""
        u = d.get("usage", {}) or {}
        pt, ct = int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
        self.db.execute("INSERT INTO usage (day, provider, model, prompt, completion, seconds, ts, key_id) VALUES (?,?,?,?,?,?,?,?)",
                        (date.today().isoformat(), self.provider, self.model, pt, ct, secs, time.time(), self.key_id))
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)",
                        (k, json.dumps({"text": text, "pt": pt, "ct": ct, "s": secs})))
        self.db.commit()
        return LLMResponse(text, pt, ct, secs, False, self.model)


def parse_json(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating code fences or leading prose."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("{"), t.rfind("}")
        if a >= 0 and b > a:
            return json.loads(t[a:b + 1])
        raise
