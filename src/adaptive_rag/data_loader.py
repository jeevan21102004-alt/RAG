from pathlib import Path

from .models import Document


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def load_documents(data_dir: Path | None = None) -> list[Document]:
    folder = data_dir or DATA_DIR
    paths = sorted([*folder.glob("*.md"), *folder.glob("*.txt")])

    documents: list[Document] = []
    for path in paths:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(Document(source=path.name, text=text))

    return documents
