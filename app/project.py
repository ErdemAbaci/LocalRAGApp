import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ENV_VAR = "LOCAL_RAG_HOME"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    docs_dir: Path
    db_path: Path
    history_path: Path
    benchmark_report_path: Path
    session_export_dir: Path


def resolve_project_root(explicit_path=None, environ=None):
    environment = os.environ if environ is None else environ
    configured_path = explicit_path

    if configured_path is None:
        configured_path = environment.get(PROJECT_ENV_VAR)

    if configured_path is None or not str(configured_path).strip():
        return DEFAULT_PROJECT_ROOT.resolve()

    candidate = Path(str(configured_path).strip()).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate

    candidate = candidate.resolve()

    if not candidate.exists():
        raise ProjectConfigurationError(
            f"Proje klasörü bulunamadı: {candidate}"
        )

    if not candidate.is_dir():
        raise ProjectConfigurationError(
            f"Proje yolu bir klasör olmalıdır: {candidate}"
        )

    return candidate


def get_project_paths(explicit_path=None, environ=None):
    root = resolve_project_root(explicit_path, environ=environ)
    return ProjectPaths(
        root=root,
        docs_dir=root / "docs",
        db_path=root / "data" / "rag.db",
        history_path=root / "data" / "cli_history",
        benchmark_report_path=root / "data" / "model_benchmark.json",
        session_export_dir=root / "data" / "exports",
    )
