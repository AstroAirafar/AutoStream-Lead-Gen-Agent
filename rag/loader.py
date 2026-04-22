from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


KB_PATH = Path(__file__).resolve().parent.parent / "knowledge_base" / "autostream_kb.md"


def load_documents() -> list[Document]:
	"""Load the markdown knowledge base and split it into embedding chunks."""
	if not KB_PATH.exists():
		raise FileNotFoundError(f"Knowledge base not found at: {KB_PATH}")

	raw_text = KB_PATH.read_text(encoding="utf-8")

	splitter = RecursiveCharacterTextSplitter(
		chunk_size=400,
		chunk_overlap=60,
		separators=["\n## ", "\n### ", "\n- ", "\n", " "],
	)

	chunks = splitter.create_documents([raw_text])
	print(f"[RAG] Loaded {len(chunks)} chunks from knowledge base.")
	return chunks
