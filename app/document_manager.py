import shutil
from pathlib import Path

from app.index_state import SUPPORTED_DOCUMENT_EXTENSIONS
from app.ingest import read_pdf_file, read_txt_file


class DocumentManagementError(ValueError):
    pass


def validate_document(file_path):
    path = Path(file_path).expanduser()

    if not path.exists():
        raise DocumentManagementError(f"Dosya bulunamadı: {path}")

    if not path.is_file():
        raise DocumentManagementError(f"Bu yol bir dosya değil: {path}")

    if path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise DocumentManagementError("Yalnızca TXT ve PDF dosyaları desteklenir.")

    try:
        if path.suffix.lower() == ".txt":
            has_text = bool(read_txt_file(path)["text"].strip())
        else:
            has_text = bool(read_pdf_file(path))
    except UnicodeDecodeError as error:
        raise DocumentManagementError("TXT dosyası UTF-8 olarak okunamıyor.") from error
    except Exception as error:
        raise DocumentManagementError(f"Dosya okunamadı: {path.name}") from error

    if not has_text:
        raise DocumentManagementError("Dosyada indekslenebilir metin bulunamadı.")

    return path


def add_document(source_path, docs_dir):
    source = validate_document(source_path)
    directory = Path(docs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / source.name

    if source.resolve() == destination.resolve():
        raise DocumentManagementError(f"{source.name} zaten docs/ klasöründe.")

    if destination.exists():
        raise DocumentManagementError(
            f"{source.name} docs/ klasöründe zaten var; mevcut dosyanın üzerine yazılmadı."
        )

    try:
        with source.open("rb") as source_file, destination.open("xb") as destination_file:
            shutil.copyfileobj(source_file, destination_file)
        validate_document(destination)
    except FileExistsError as error:
        raise DocumentManagementError(
            f"{source.name} docs/ klasöründe zaten var; mevcut dosyanın üzerine yazılmadı."
        ) from error
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    return destination


def resolve_managed_document(source_name, docs_dir):
    name = str(source_name).strip()
    candidate = Path(name)

    if not name or candidate.name != name or candidate.is_absolute():
        raise DocumentManagementError(
            "Silmek için docs/ içindeki dosyanın yalnızca adını yaz."
        )

    if candidate.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise DocumentManagementError("Yalnızca TXT ve PDF dosyaları yönetilebilir.")

    destination = Path(docs_dir) / name

    if not destination.is_file():
        raise DocumentManagementError(f"docs/ içinde {name} bulunamadı.")

    return destination


def remove_document(source_name, docs_dir):
    destination = resolve_managed_document(source_name, docs_dir)
    destination.unlink()
    return destination
