from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import build_graph
from agent.state import AgentState
from rag.loader import load_documents
from rag.retriever import build_retriever

load_dotenv()


def get_initial_state() -> AgentState:
	return {
		"messages": [],
		"intent": "greeting",
		"lead_name": None,
		"lead_email": None,
		"lead_platform": None,
		"lead_captured": False,
		"awaiting_lead_field": None,
	}


def get_last_ai_message(state: AgentState) -> str:
	ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
	if not ai_messages:
		return "Sorry, I couldn't generate a response."
	return ai_messages[-1].content


def main() -> None:
	print("\n" + "=" * 60)
	print("  AutoStream AI Assistant")
	print("  Powered by Groq + LangGraph + FAISS RAG")
	print("=" * 60)
	print("  Type 'quit' or 'exit' to end the conversation.\n")

	print("[STARTUP] Loading knowledge base...")
	documents = load_documents()
	retriever = build_retriever(documents)

	print("[STARTUP] Compiling agent graph...")
	graph = build_graph(retriever)
	print("[STARTUP] Ready!\n")

	state = get_initial_state()
	print(
		"AutoStream Assistant: Hi! I'm the AutoStream assistant. "
		"I can help you with pricing, features, or getting started. "
		"What can I help you with today?\n"
	)

	try:
		while True:
			user_input = input("Client: ").strip()

			if not user_input:
				continue

			if user_input.lower() in ("quit", "exit", "bye"):
				print("\nAutoStream Assistant: Thanks for chatting! Have a great day.")
				break

			state["messages"].append(HumanMessage(content=user_input))
			state = graph.invoke(state)
			print(f"\nAutoStream Assistant: {get_last_ai_message(state)}\n")

	except KeyboardInterrupt:
		print("\n\nAutoStream Assistant: Goodbye!")


if __name__ == "__main__":
	main()
