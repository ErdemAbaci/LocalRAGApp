import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from app import database
from app.index_state import get_index_freshness
from app.ingest import DOCS_DIR
from app.llm import MODEL_ALIAS


EXPECTED_EMBEDDING_DIMENSION = 384
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".txt"}


@dataclass(frozen=True)
class HealthCheck:
    name: str
    status: str
    message: str
    solution: str | None = None


def check_documents(docs_dir=None):
    directory = Path(docs_dir) if docs_dir is not None else DOCS_DIR

    if not directory.is_dir():
        return HealthCheck(
            name="Dokümanlar",
            status="error",
            message=f"{directory} klasörü bulunamadı.",
            solution=f"{directory} klasörünü oluştur ve içine TXT veya PDF ekle.",
        )

    documents = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    ]

    if not documents:
        return HealthCheck(
            name="Dokümanlar",
            status="warning",
            message="İndekslenebilir TXT veya PDF bulunamadı.",
            solution=f"{directory} klasörüne en az bir TXT veya PDF ekle.",
        )

    return HealthCheck(
        name="Dokümanlar",
        status="ok",
        message=f"{len(documents)} desteklenen dosya hazır.",
    )


def check_index(db_path=None):
    path = Path(db_path) if db_path is not None else database.DB_PATH

    if not path.exists():
        missing_database = HealthCheck(
            name="Veritabanı",
            status="warning",
            message=f"{path} henüz oluşturulmamış.",
            solution="/reindex çalıştır.",
        )
        missing_index = HealthCheck(
            name="Embedding indeksi",
            status="warning",
            message="Kontrol edilecek indeks bulunamadı.",
            solution="/reindex çalıştır.",
        )
        return [missing_database, missing_index]

    try:
        chunks = database.get_all_chunks()
    except Exception as error:
        return [
            HealthCheck(
                name="Veritabanı",
                status="error",
                message=f"İndeks okunamadı: {error}",
                solution="Önce /reindex çalıştır; sorun sürerse data/rag.db dosyasını kontrol et.",
            ),
            HealthCheck(
                name="Embedding indeksi",
                status="warning",
                message="Veritabanı hatası nedeniyle kontrol edilemedi.",
                solution="Veritabanı hatasını gider ve /doctor komutunu yeniden çalıştır.",
            ),
        ]

    if not chunks:
        return [
            HealthCheck(
                name="Veritabanı",
                status="ok",
                message=f"{path} okunabiliyor.",
            ),
            HealthCheck(
                name="Embedding indeksi",
                status="warning",
                message="İndeks boş.",
                solution="/reindex çalıştır.",
            ),
        ]

    source_count = len({chunk["source_name"] for chunk in chunks})
    database_check = HealthCheck(
        name="Veritabanı",
        status="ok",
        message=f"{source_count} kaynak ve {len(chunks)} chunk okunabiliyor.",
    )

    for chunk in chunks:
        embedding = chunk.get("embedding")

        if not isinstance(embedding, list) or len(embedding) != EXPECTED_EMBEDDING_DIMENSION:
            return [
                database_check,
                HealthCheck(
                    name="Embedding indeksi",
                    status="error",
                    message=f"chunk_id={chunk['id']} embedding boyutu geçersiz.",
                    solution="/reindex çalıştır.",
                ),
            ]

        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in embedding):
            return [
                database_check,
                HealthCheck(
                    name="Embedding indeksi",
                    status="error",
                    message=f"chunk_id={chunk['id']} geçersiz embedding değeri içeriyor.",
                    solution="/reindex çalıştır.",
                ),
            ]

    return [
        database_check,
        HealthCheck(
            name="Embedding indeksi",
            status="ok",
            message=(
                f"{len(chunks)} embedding sağlıklı "
                f"({EXPECTED_EMBEDDING_DIMENSION} boyut)."
            ),
        ),
    ]


def check_index_freshness(docs_dir=None, db_path=None):
    directory = Path(docs_dir) if docs_dir is not None else DOCS_DIR
    path = Path(db_path) if db_path is not None else database.DB_PATH
    freshness = get_index_freshness(directory, path)

    if freshness.status == "current":
        return HealthCheck(
            name="İndeks güncelliği",
            status="ok",
            message="Dokümanlar indeksle eşleşiyor.",
        )

    if freshness.status == "stale":
        return HealthCheck(
            name="İndeks güncelliği",
            status="warning",
            message=f"İndeks güncel değil. {freshness.change_summary()}",
            solution="/reindex veya local-rag reindex çalıştır.",
        )

    if freshness.status == "untracked":
        return HealthCheck(
            name="İndeks güncelliği",
            status="warning",
            message="İndeksin doküman özeti bulunamadı.",
            solution="/reindex veya local-rag reindex çalıştır.",
        )

    if freshness.status == "missing":
        return HealthCheck(
            name="İndeks güncelliği",
            status="warning",
            message="Kontrol edilecek indeks bulunamadı.",
            solution="/reindex veya local-rag reindex çalıştır.",
        )

    return HealthCheck(
        name="İndeks güncelliği",
        status="error",
        message=f"Doküman değişiklikleri kontrol edilemedi: {freshness.error}",
        solution="Dosya izinlerini kontrol et ve /doctor komutunu yeniden çalıştır.",
    )


def check_foundry(foundry_home=None, executable_finder=shutil.which):
    if executable_finder("foundry") is None:
        return [
            HealthCheck(
                name="Foundry Local",
                status="error",
                message="foundry terminal komutu bulunamadı.",
                solution="Foundry Local kurulumunu kontrol et.",
            ),
            HealthCheck(
                name="LLM modeli",
                status="warning",
                message=f"{MODEL_ALIAS} cache durumu kontrol edilemedi.",
                solution="Önce Foundry Local kurulumunu tamamla.",
            ),
        ]

    home = Path(foundry_home) if foundry_home is not None else Path.home() / ".foundry"
    config_path = home / "foundry.config.json"

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        cache_directory = config["serviceSettings"]["cacheDirectoryPath"]
        cache_path = Path(cache_directory).expanduser()
    except Exception as error:
        return [
            HealthCheck(
                name="Foundry Local",
                status="error",
                message=f"Cache yapılandırması okunamadı: {error}",
                solution="Foundry Local kurulumunu kontrol et.",
            ),
            HealthCheck(
                name="LLM modeli",
                status="warning",
                message=f"{MODEL_ALIAS} cache durumu kontrol edilemedi.",
                solution="Foundry Local sorununu gider ve /doctor komutunu yeniden çalıştır.",
            ),
        ]

    if not cache_path.is_dir():
        return [
            HealthCheck(
                name="Foundry Local",
                status="error",
                message=f"Model cache dizini bulunamadı: {cache_path}",
                solution="Foundry Local kurulumunu kontrol et.",
            ),
            HealthCheck(
                name="LLM modeli",
                status="warning",
                message=f"{MODEL_ALIAS} cache durumu kontrol edilemedi.",
                solution=f"Gerekirse foundry model download {MODEL_ALIAS} çalıştır.",
            ),
        ]

    foundry_check = HealthCheck(
        name="Foundry Local",
        status="ok",
        message="Terminal aracı ve model cache dizini hazır.",
    )
    model_name = MODEL_ALIAS.lower()
    model_cached = any(
        model_name in str(metadata_path.parent).lower()
        and any(
            model_file.is_file() and model_file.stat().st_size > 0
            for model_file in metadata_path.parent.glob("model.onnx*")
        )
        for metadata_path in cache_path.rglob("inference_model.json")
    )

    if not model_cached:
        return [
            foundry_check,
            HealthCheck(
                name="LLM modeli",
                status="error",
                message=f"{MODEL_ALIAS} yerel cache içinde bulunamadı.",
                solution=f"foundry model download {MODEL_ALIAS} çalıştır.",
            ),
        ]

    return [
        foundry_check,
        HealthCheck(
            name="LLM modeli",
            status="ok",
            message=f"{MODEL_ALIAS} cache içinde hazır.",
        ),
    ]


def run_health_checks(
    docs_dir=None,
    db_path=None,
    foundry_home=None,
    executable_finder=shutil.which,
):
    checks = [check_documents(docs_dir=docs_dir)]
    checks.append(check_index_freshness(docs_dir=docs_dir, db_path=db_path))
    checks.extend(check_index(db_path=db_path))
    checks.extend(
        check_foundry(
            foundry_home=foundry_home,
            executable_finder=executable_finder,
        )
    )
    return checks
