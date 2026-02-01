from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from tqdm import tqdm
import os

print("🚀 Indexador iniciado")

# ---------------------------
# CONFIG
# ---------------------------
DOCS_PATH = "docs/markdown"
DB_PATH = "db"
BATCH_SIZE = 25

# ---------------------------
# LOAD DOCUMENTS
# ---------------------------
loader = DirectoryLoader(
    DOCS_PATH,
    glob="**/*.md",
    loader_cls=TextLoader,
    show_progress=True
)

documents = loader.load()
print(f"📄 {len(documents)} documentos carregados")

# ---------------------------
# SPLIT DOCUMENTS
# ---------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(documents)
print(f"🧩 {len(splits)} chunks gerados")

# ---------------------------
# EMBEDDINGS
# ---------------------------
print("🧠 Inicializando embeddings")
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# ---------------------------
# VECTOR STORE
# ---------------------------
print("📦 Criando / abrindo Chroma")
vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

# ---------------------------
# INDEXAÇÃO EM BATCHES
# ---------------------------
print("📦 Iniciando indexação em batches")

for i in range(0, len(splits), BATCH_SIZE):
    batch = splits[i:i + BATCH_SIZE]
    print(f"🔄 Indexando chunks {i} → {i + len(batch)}")
    vectorstore.add_documents(batch)


print("✅ Indexação concluída com sucesso")
