from __future__ import annotations

import json
from pathlib import Path


class ByteTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = max(1, int(vocab_size))

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8") or b"\x00"
        return [int(byte) % self.vocab_size for byte in data]

    def decode(self, token_ids: list[int]) -> str:
        chars = []
        for token_id in token_ids:
            value = int(token_id)
            if 32 <= value <= 126:
                chars.append(chr(value))
            else:
                chars.append(f"<{value}>")
        return "".join(chars)


class TokenizerBridge:
    def __init__(self, model_path: str | Path, vocab_size: int):
        self.model_path = Path(model_path)
        self.backend = self._load_backend(vocab_size)

    def _load_backend(self, vocab_size: int):
        tokenizer_json = self.model_path / "tokenizer.json"
        if tokenizer_json.exists():
            try:
                from tokenizers import Tokenizer

                return Tokenizer.from_file(str(tokenizer_json))
            except ImportError:
                pass
        config_path = self.model_path / "tokenizer_config.json"
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
                if data.get("type") == "byte_mod":
                    return ByteTokenizer(int(data.get("vocab_size", vocab_size)))
            except json.JSONDecodeError:
                pass
        return ByteTokenizer(vocab_size)

    def encode(self, text: str) -> list[int]:
        if hasattr(self.backend, "encode") and self.backend.__class__.__name__ == "Tokenizer":
            return self.backend.encode(text).ids
        return self.backend.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        if hasattr(self.backend, "decode") and self.backend.__class__.__name__ == "Tokenizer":
            return self.backend.decode(token_ids)
        return self.backend.decode(token_ids)
