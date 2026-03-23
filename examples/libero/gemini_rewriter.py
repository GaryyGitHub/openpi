from __future__ import annotations

import dataclasses
import json
import os
import re
import urllib.error
import urllib.request

_STOPWORDS = {
    "the",
    "and",
    "then",
    "with",
    "from",
    "into",
    "onto",
    "that",
    "this",
    "task",
    "please",
    "scene",
    "setup",
    "object",
    "objects",
    "while",
    "only",
    "keep",
    "all",
    "other",
}


@dataclasses.dataclass
class GeminiConfig:
    api_key: str = ""
    api_base: str = "https://api2.xcodecli.com/"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.2
    max_output_tokens: int = 128
    timeout_sec: int = 20


class GeminiInstructionRefiner:
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.api_key = config.api_key.strip() or os.getenv("GEMINI_API_KEY", "").strip()
        self.api_base = (
            config.api_base.strip() or os.getenv("GEMINI_API_BASE", "").strip() or "https://api2.xcodecli.com"
        ).rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def refine(self, original: str, stage1_text: str, variant: str) -> tuple[str, str]:
        if not self.enabled:
            return stage1_text, "disabled"

        prompt = self._build_prompt(original=original, stage1_text=stage1_text, variant=variant)
        try:
            candidate = self._call_gemini(prompt)
        except Exception:
            return stage1_text, "api_error"

        if not candidate:
            return stage1_text, "empty"

        if not _passes_safety_gate(original, candidate):
            return stage1_text, "rejected"

        return candidate, "accepted"

    def _call_gemini(self, prompt: str) -> str:
        candidate_bases = [self.api_base]
        if self.api_base.endswith("api.xcodecli.com"):
            candidate_bases.append("https://api2.xcodecli.com")

        last_error: Exception | None = None
        for base in candidate_bases:
            try:
                return self._call_gemini_once(base, prompt)
            except RuntimeError as exc:
                last_error = exc
                if "403" in str(exc) and "1010" in str(exc) and base.endswith("api.xcodecli.com"):
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini request failed with unknown error")

    def _call_gemini_once(self, base: str, prompt: str) -> str:
        url = f"{base}/v1beta/models/{self.config.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
                "responseMimeType": "text/plain",
            },
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Gemini HTTPError ({base}): {exc.code} {detail}") from exc

        data = json.loads(body)
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return ""
        text = parts[0].get("text", "")
        return _clean_output(text)

    def _build_prompt(self, original: str, stage1_text: str, variant: str) -> str:
        variant_spec = {
            "paraphrase": "同义改写, 保留任务目标、对象和空间关系。",
            "constraint": "加入轻量过程约束(不改变目标)。",
            "reference": "加入上下文或指代表达(不改变目标)。",
            "noisy": "加入自然口语噪声词(不改变目标)。",
        }.get(variant, "保持语义不变进行自然改写。")

        return (
            "You are rewriting robot manipulation instructions for LIBERO benchmark.\n"
            "Return exactly one line in English.\n"
            "Hard constraints:\n"
            "1) Preserve original task semantics and goal exactly.\n"
            "2) Keep key object names and spatial relations.\n"
            "3) No extra sub-goals, no negation flips, no ambiguity increase.\n"
            f"Rewrite style: {variant_spec}\n\n"
            f"Original instruction: {original}\n"
            f"Stage-1 rewrite: {stage1_text}\n"
            "Output only the final rewritten instruction."
        )


def _clean_output(text: str) -> str:
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    line = line.strip('"').strip("'")
    return re.sub(r"\s+", " ", line).strip().rstrip(".")


def _passes_safety_gate(original: str, rewritten: str) -> bool:
    if not rewritten:
        return False
    if len(rewritten) < 12:
        return False

    original_tokens = _keywords(original)
    if not original_tokens:
        return True
    rewritten_l = rewritten.lower()
    overlap = sum(1 for token in original_tokens if token in rewritten_l)
    return overlap >= max(1, min(3, len(original_tokens) // 3))


def _keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]+", text.lower())
    tokens = [t for t in tokens if len(t) >= 4 and t not in _STOPWORDS]
    out: list[str] = []
    for token in tokens:
        if token not in out:
            out.append(token)
    return out[:12]
