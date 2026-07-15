import hashlib
from dataclasses import dataclass
from pathlib import Path

from app import database


SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".txt"}
HASH_BLOCK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class IndexFreshness:
    status: str
    added: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    error: str | None = None

    @property
    def is_current(self):
        return self.status == "current"

    def change_summary(self):
        parts = []

        if self.added:
            parts.append(f"Eklenen: {', '.join(self.added)}")

        if self.modified:
            parts.append(f"Değişen: {', '.join(self.modified)}")

        if self.deleted:
            parts.append(f"Silinen: {', '.join(self.deleted)}")

        return " · ".join(parts)

    def display_status(self):
        if self.status == "current":
            return "güncel"

        if self.status == "stale":
            return f"güncel değil · {self.change_summary()}"

        if self.status == "untracked":
            return "bilinmiyor · yeniden indeks gerekli"

        if self.status == "missing":
            return "indeks bulunamadı"

        return f"kontrol edilemedi · {self.error}"


def list_document_paths(docs_dir):
    directory = Path(docs_dir)

    if not directory.is_dir():
        return []

    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
        ),
        key=lambda path: path.name,
    )


def hash_file(file_path):
    digest = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(HASH_BLOCK_SIZE), b""):
            digest.update(block)

    return digest.hexdigest()


def build_source_manifest(docs_dir):
    return [
        {
            "source_name": file_path.name,
            "source_type": file_path.suffix.lower().lstrip("."),
            "file_size": file_path.stat().st_size,
            "sha256": hash_file(file_path),
        }
        for file_path in list_document_paths(docs_dir)
    ]


def get_index_freshness(docs_dir, db_path=None):
    path = Path(db_path) if db_path is not None else database.DB_PATH

    if not path.exists():
        return IndexFreshness(status="missing")

    try:
        current_manifest = build_source_manifest(docs_dir)
        stored_manifest = database.get_source_manifest(db_path=path)
    except Exception as error:
        return IndexFreshness(status="error", error=str(error))

    if not stored_manifest:
        return IndexFreshness(status="untracked")

    current_by_name = {
        source["source_name"]: source
        for source in current_manifest
    }
    stored_by_name = {
        source["source_name"]: source
        for source in stored_manifest
    }

    current_names = set(current_by_name)
    stored_names = set(stored_by_name)
    added = tuple(sorted(current_names - stored_names))
    deleted = tuple(sorted(stored_names - current_names))
    modified = tuple(sorted(
        source_name
        for source_name in current_names & stored_names
        if current_by_name[source_name] != stored_by_name[source_name]
    ))

    if added or modified or deleted:
        return IndexFreshness(
            status="stale",
            added=added,
            modified=modified,
            deleted=deleted,
        )

    return IndexFreshness(status="current")
