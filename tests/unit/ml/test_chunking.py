import itertools

import pytest

from edu_graphrag.ml.ingest.chunking import ChunkingConfig, chunk_text, split_sections

WORDS = " ".join(f"word{i:04d}" for i in range(600))  # ~5400 chars, no headings


def test_split_sections_by_headings_ignoring_code_fences() -> None:
    text = "intro\n# A\naaa\n```python\n# comment, not a heading\n```\n## B\nbbb\n\n"
    sections = split_sections(text)
    assert sections == [
        "intro",
        "# A\naaa\n```python\n# comment, not a heading\n```",
        "## B\nbbb",
    ]
    assert split_sections("") == []
    assert split_sections("#hashtag is not a heading") == ["#hashtag is not a heading"]


def test_small_sections_are_packed_together() -> None:
    cfg = ChunkingConfig(target_chars=300, overlap_chars=50, min_chars=10)
    text = "\n".join(f"# H{i}\n" + "x " * 30 for i in range(6))
    chunks = chunk_text(text, cfg)
    assert 1 < len(chunks) < 6
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0].startswith("# H0")


def test_long_section_windows_overlap_and_respect_words() -> None:
    cfg = ChunkingConfig(target_chars=1200, overlap_chars=150, min_chars=80)
    chunks = chunk_text(WORDS, cfg)
    assert len(chunks) >= 5
    words = set(WORDS.split())
    for chunk in chunks:
        assert len(chunk) <= 1200
        # Every token is a whole word: no cut inside a word.
        assert set(chunk.split()) <= words
    # Consecutive windows overlap.
    for left, right in itertools.pairwise(chunks):
        assert right.split()[0] in left.split()


def test_unbreakable_text_is_hard_cut() -> None:
    cfg = ChunkingConfig(target_chars=100, overlap_chars=0, min_chars=1)
    chunks = chunk_text("x" * 250, cfg)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_short_and_blank_chunks_are_dropped() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []
    assert chunk_text("# Title\n\ntoo short") == []
    assert chunk_text("a" * 100) == ["a" * 100]


def test_chunking_is_deterministic() -> None:
    assert chunk_text(WORDS) == chunk_text(WORDS)


@pytest.mark.parametrize(("target", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_config(target: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="chars"):
        ChunkingConfig(target_chars=target, overlap_chars=overlap)
