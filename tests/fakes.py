"""Offline stand-ins for the OpenAI and Anthropic clients, plus the fixture loader.

The fakes return real SDK response objects and reject keyword arguments the real SDK
method does not accept, so a typo in a call site fails here rather than in production.
Nothing in the test suite touches the network.
"""

import inspect
import json
import re
import zlib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

from anthropic.resources.messages import Messages
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from openai.resources.chat.completions import Completions
from openai.resources.embeddings import Embeddings
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion

FIXTURES = Path(__file__).parent / "fixtures"


def load_cases(name: str) -> list[dict]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def case_ids(cases: list[dict]) -> list[str]:
    return [c["id"] for c in cases]


def _check_kwargs(method, kwargs: dict) -> None:
    unknown = set(kwargs) - set(inspect.signature(method).parameters)
    assert not unknown, f"{method.__qualname__} does not accept {sorted(unknown)}"


def chat_completion(text: str | None) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
        }
    )


def word_vector(text: str, dims: int = 512) -> list[float]:
    """Deterministic bag-of-words embedding: similar wording gives similar vectors."""
    vec = [0.0] * dims
    for word in re.findall(r"\w+", text.lower()):
        vec[zlib.crc32(word.encode()) % dims] += 1.0
    return vec


class FakeOpenAI:
    """Replies to chat completions with `reply`; embeddings come from `embed`."""

    def __init__(self, reply: str | None = "", embed: Callable[[str], list[float]] = word_vector):
        self.reply, self.embed = reply, embed
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.embeddings = SimpleNamespace(create=self._embeddings)

    def _chat(self, **kwargs):
        _check_kwargs(Completions.create, kwargs)
        self.calls.append(kwargs)
        return chat_completion(self.reply)

    def _embeddings(self, **kwargs):
        _check_kwargs(Embeddings.create, kwargs)
        self.calls.append(kwargs)
        return CreateEmbeddingResponse.model_validate(
            {
                "object": "list",
                "model": "fake",
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
                "data": [
                    {"object": "embedding", "index": i, "embedding": self.embed(text)}
                    for i, text in enumerate(kwargs["input"])
                ],
            }
        )


def text_block(text: str) -> TextBlock:
    return TextBlock(type="text", text=text)


def tool_use(name: str, **input) -> ToolUseBlock:
    return ToolUseBlock(type="tool_use", id="toolu_fake", name=name, input=input)


def chat_completion_from_blocks(blocks) -> ChatCompletion:
    """Anthropic content blocks as the OpenAI chat completion that a call site
    migrated by Offramp reads: text blocks joined, tool_use blocks as tool calls."""
    text = "".join(b.text for b in blocks if b.type == "text")
    calls = [
        {
            "id": b.id,
            "type": "function",
            "function": {"name": b.name, "arguments": json.dumps(b.input)},
        }
        for b in blocks
        if b.type == "tool_use"
    ]
    message = {"role": "assistant", "content": text or None}
    if calls:
        message["tool_calls"] = calls
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if calls else "stop",
                    "message": message,
                }
            ],
        }
    )


class FakeAnthropic:
    """Replies to messages with the given content blocks (a bare string becomes a text block)."""

    def __init__(self, *blocks: TextBlock | ToolUseBlock | str):
        self.blocks = [text_block(b) if isinstance(b, str) else b for b in blocks]
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._messages)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat_completion))

    def _messages(self, **kwargs):
        _check_kwargs(Messages.create, kwargs)
        self.calls.append(kwargs)
        return Message(
            id="msg_fake",
            type="message",
            role="assistant",
            model="fake",
            content=self.blocks,
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage(input_tokens=0, output_tokens=0),
        )

    def _chat_completion(self, **kwargs):
        """The same canned reply, as an OpenAI chat completion (Offramp)."""
        _check_kwargs(Completions.create, kwargs)
        self.calls.append(kwargs)
        return chat_completion_from_blocks(self.blocks)
