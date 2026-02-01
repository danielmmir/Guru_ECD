from docling.document_converter import DocumentConverter
from pathlib import Path

# Caminhos do projeto
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
OUTPUT_DIR = DOCS_DIR / "markdown"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Arquivo de entrada
INPUT_FILE = DOCS_DIR / "Manual_ECD.docx"

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_FILE}")

print(f"📄 Convertendo {INPUT_FILE.name} para Markdown...")

converter = DocumentConverter()
result = converter.convert(INPUT_FILE)

output_path = OUTPUT_DIR / "Manual_ECD.md"
output_path.write_text(
    result.document.export_to_markdown(),
    encoding="utf-8"
)

print(f"✅ Markdown gerado com sucesso em: {output_path}")
