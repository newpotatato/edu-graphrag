"""Source connectors: where raw study documents come from.

`Connector` is the extension point for new sources. The filesystem connector below reads a
local archive; future connectors (YouTube transcripts, an LMS export, a Kafka topic used as a
transport) only have to yield `RawDocument` objects and the rest of the pipeline (conversion,
chunking, dataset build) stays unchanged.
"""

import hashlib
import re
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

import structlog

logger = structlog.get_logger(__name__)

SourceType = Literal["pdf", "md", "ipynb", "tex", "docx", "pptx"]

# File extension (lower case, with dot) -> source type.
SUPPORTED_EXTENSIONS: dict[str, SourceType] = {
    ".pdf": "pdf",
    ".md": "md",
    ".ipynb": "ipynb",
    ".tex": "tex",
    ".docx": "docx",
    ".pptx": "pptx",
}

# Directory names skipped anywhere in the tree (hidden directories are skipped as well).
DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {"node_modules", ".venv", "venv", ".git", "__pycache__", ".ipynb_checkpoints"}
)
# Paths relative to the root that are skipped entirely (this repo lives there).
DEFAULT_EXCLUDE_PREFIXES: tuple[str, ...] = ("3_grade/MLops",)

_GRADE_RE = re.compile(r"^(\d+)_grade$")
_SEMESTER_RE = re.compile(r"^(\d+)_sem$")


@dataclass(frozen=True, slots=True)
class PathInfo:
    """Metadata derived from the location of a file in the archive."""

    grade: int | None
    semester: int | None
    subject: str | None
    is_sorted: bool


def parse_path(source_uri: str | PurePosixPath) -> PathInfo:
    """Derive grade/semester/subject from a path relative to the archive root.

    Sorted layout: `<g>_grade/<s>_sem/<subject>/.../file`. Unsorted layout (new data without a
    semester level): `<g>_grade/<subject>/.../file`. Subject folder names are kept as-is.
    """
    parts = PurePosixPath(source_uri).parts
    dirs = parts[:-1]
    if not dirs or (grade_match := _GRADE_RE.match(dirs[0])) is None:
        return PathInfo(grade=None, semester=None, subject=None, is_sorted=False)
    grade = int(grade_match.group(1))
    if len(dirs) < 2:
        return PathInfo(grade=grade, semester=None, subject=None, is_sorted=False)
    if (sem_match := _SEMESTER_RE.match(dirs[1])) is not None:
        semester = int(sem_match.group(1))
        if len(dirs) < 3:
            return PathInfo(grade=grade, semester=semester, subject=None, is_sorted=False)
        return PathInfo(grade=grade, semester=semester, subject=dirs[2], is_sorted=True)
    return PathInfo(grade=grade, semester=None, subject=dirs[1], is_sorted=False)


def is_excluded(
    source_uri: str | PurePosixPath,
    exclude_dirs: Collection[str] = DEFAULT_EXCLUDE_DIRS,
    exclude_prefixes: Collection[str] = DEFAULT_EXCLUDE_PREFIXES,
) -> bool:
    """True if the relative path lies in an excluded or hidden directory or under a prefix."""
    path = PurePosixPath(source_uri)
    for prefix in exclude_prefixes:
        if path == PurePosixPath(prefix) or path.is_relative_to(prefix):
            return True
    return any(part in exclude_dirs or part.startswith(".") for part in path.parts[:-1])


def make_doc_id(source_uri: str) -> str:
    """Stable document id: first 16 hex chars of sha1 of the relative path."""
    return hashlib.sha1(source_uri.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def source_type_for(path: str | PurePosixPath | Path) -> SourceType | None:
    """Source type by file extension, None if the file is not supported."""
    return SUPPORTED_EXTENSIONS.get(PurePosixPath(path).suffix.lower())


@dataclass(frozen=True, slots=True)
class RawDocument:
    """A source document before conversion."""

    source_uri: str  # posix path relative to the source root
    source_type: SourceType
    content_hash: str  # sha256 of the raw bytes
    grade: int | None
    semester: int | None
    subject: str | None
    is_sorted: bool
    path: Path  # local file to read; remote connectors materialise to a temp file

    @property
    def doc_id(self) -> str:
        return make_doc_id(self.source_uri)

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()


class Connector(Protocol):
    """Yields raw documents from some source (filesystem, YouTube, LMS, Kafka, ...)."""

    def iter_documents(self) -> Iterator[RawDocument]:
        """Yield every supported document, in a deterministic order."""
        ...


class FilesystemConnector:
    """Walks a local archive laid out as `<grade>_grade/[<sem>_sem/]<subject>/...`."""

    def __init__(
        self,
        root: Path,
        exclude_dirs: Collection[str] = DEFAULT_EXCLUDE_DIRS,
        exclude_prefixes: Collection[str] = DEFAULT_EXCLUDE_PREFIXES,
    ) -> None:
        self.root = root
        self.exclude_dirs = frozenset(exclude_dirs)
        self.exclude_prefixes = tuple(exclude_prefixes)

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _skip_dir(self, path: Path) -> bool:
        if path.name.startswith(".") or path.name in self.exclude_dirs:
            return True
        rel = PurePosixPath(self._relative(path))
        return any(rel == PurePosixPath(p) or rel.is_relative_to(p) for p in self.exclude_prefixes)

    def iter_paths(self) -> Iterator[Path]:
        """Supported files under the root, sorted by relative path."""
        found: list[Path] = []
        for dirpath, dirnames, filenames in self.root.walk():
            # Prune excluded directories in place so we never descend into them.
            dirnames[:] = [name for name in dirnames if not self._skip_dir(dirpath / name)]
            for name in filenames:
                # Hidden files and Office lock files (~$foo.docx).
                if name.startswith((".", "~$")) or source_type_for(name) is None:
                    continue
                found.append(dirpath / name)
        found.sort(key=self._relative)
        yield from found

    def iter_documents(self) -> Iterator[RawDocument]:
        for path in self.iter_paths():
            source_uri = self._relative(path)
            source_type = source_type_for(path)
            if source_type is None:  # pragma: no cover - already filtered while walking
                continue
            info = parse_path(source_uri)
            yield RawDocument(
                source_uri=source_uri,
                source_type=source_type,
                content_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
                grade=info.grade,
                semester=info.semester,
                subject=info.subject,
                is_sorted=info.is_sorted,
                path=path,
            )
