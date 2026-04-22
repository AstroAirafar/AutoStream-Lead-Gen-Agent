from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_retriever(documents: list[Document]):
	"""Build a FAISS-backed retriever once at startup."""
	print(f"[RAG] Building FAISS index with model: {EMBEDDING_MODEL}")
	embeddings = HuggingFaceEmbeddings(
		model_name=EMBEDDING_MODEL,
		model_kwargs={"device": "cpu"},
		encode_kwargs={"normalize_embeddings": True},
	)
	vector_store = FAISS.from_documents(documents, embeddings)
	retriever = vector_store.as_retriever(
		search_type="similarity",
		search_kwargs={"k": 2},
	)
	print("[RAG] FAISS index ready.")
	return retriever


def retrieve(query: str, retriever) -> str:
	"""Retrieve top-2 relevant chunks and join them into one context string."""
	docs = retriever.invoke(query)
	if not docs:
		return "No relevant information found in the knowledge base."
	return "\n\n".join(doc.page_content for doc in docs)
