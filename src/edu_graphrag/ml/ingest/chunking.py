"""Markdown chunking: split by headings, then pack sections into size-bounded chunks.

Small consecutive sections are merged up to `target_chars`; a section longer than that is cut
into overlapping windows that end on whitespace whenever possible.
"""

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class ChunkingConfig:
    target_chars: int = 1200
    overlap_chars: int = 150
    min_chars: int = 80  # chunks with fewer non-whitespace chars are dropped

    def __post_init__(self) -> None:
        if self.target_chars <= 0:
            raise ValueError("target_chars must be positive")
        if not 0 <= self.overlap_chars < self.target_chars:
            raise ValueError("overlap_chars must be in [0, target_chars)")


def split_sections(markdown: str) -> list[str]:
    """Split text before every markdown heading; headings inside code fences are ignored."""
    sections: list[list[str]] = [[]]
    in_fence = False
    for line in markdown.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence and _HEADING_RE.match(line) and sections[-1]:
            sections.append([])
        sections[-1].append(line)
    joined = ("\n".join(lines).strip() for lines in sections)
    return [section for section in joined if section]


def _split_long(text: str, target: int, overlap: int) -> list[str]:
    """Sliding windows of at most `target` chars, cut on whitespace when possible."""
    windows: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + target, n)
        if end < n:
            # Back off to the last whitespace in the second half of the window.
            cut = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if cut > start + target // 2:
                end = cut
        window = text[start:end].strip()
        if window:
            windows.append(window)
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        # Do not start in the middle of a word: move to the next whitespace if it is near.
        if next_start > 0 and not text[next_start - 1].isspace():
            boundary = re.search(r"\s", text[next_start:end])
            if boundary is not None:
                next_start += boundary.end()
        start = next_start
    return windows


def _non_ws_len(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def chunk_text(text: str, config: ChunkingConfig | None = None) -> list[str]:
    """Chunk markdown text. Pure and deterministic."""
    cfg = config or ChunkingConfig()
    pieces: list[str] = []
    for section in split_sections(text):
        if len(section) > cfg.target_chars:
            pieces.extend(_split_long(section, cfg.target_chars, cfg.overlap_chars))
        else:
            pieces.append(section)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 2 + len(piece) <= cfg.target_chars:
            current = f"{current}\n\n{piece}"
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if _non_ws_len(chunk) >= cfg.min_chars]
