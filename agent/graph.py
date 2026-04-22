from functools import partial

from langgraph.graph import END, StateGraph

from agent.nodes import general_node, intent_node, lead_collect_node, rag_node, tool_node
from agent.state import AgentState


def route_after_intent(state: AgentState) -> str:
	"""Route from intent classification into the correct handling node."""
	intent = state.get("intent", "greeting")
	awaiting = state.get("awaiting_lead_field")
	lead_captured = state.get("lead_captured", False)

	if lead_captured:
		if intent == "inquiry":
			return "rag_node"
		return "general_node"

	if awaiting is not None or intent == "high_intent":
		return "lead_collect_node"

	if intent == "inquiry":
		return "rag_node"

	return "general_node"


def route_after_lead(state: AgentState):
	"""Route from lead collection to tool execution only when ready."""
	if state.get("awaiting_lead_field") == "__ready__":
		return "tool_node"
	return END


def build_graph(retriever):
	"""Build and compile the stateful conversational graph."""
	graph = StateGraph(AgentState)

	graph.add_node("intent_node", intent_node)
	graph.add_node("rag_node", partial(rag_node, retriever=retriever))
	graph.add_node("lead_collect_node", lead_collect_node)
	graph.add_node("tool_node", tool_node)
	graph.add_node("general_node", general_node)

	graph.set_entry_point("intent_node")

	graph.add_conditional_edges(
		"intent_node",
		route_after_intent,
		{
			"rag_node": "rag_node",
			"lead_collect_node": "lead_collect_node",
			"general_node": "general_node",
		},
	)

	graph.add_conditional_edges(
		"lead_collect_node",
		route_after_lead,
		{
			"tool_node": "tool_node",
			END: END,
		},
	)

	graph.add_edge("rag_node", END)
	graph.add_edge("tool_node", END)
	graph.add_edge("general_node", END)

	return graph.compile()
