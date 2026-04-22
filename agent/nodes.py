import os
import re

from dotenv import load_dotenv
from groq import Groq
from langchain_core.messages import AIMessage, HumanMessage

from agent.intent import classify_intent
from agent.state import AgentState
from agent.tools import mock_lead_capture
from rag.retriever import retrieve

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
_client: Groq | None = None
POST_CAPTURE_SAFE_RESPONSE = (
	"Your details have already been submitted. Our team will contact you soon. "
	"You can still ask about pricing, features, or policies."
)


def _get_client() -> Groq:
	global _client

	if _client is None:
		api_key = os.getenv("GROQ_API_KEY")
		if not api_key:
			raise RuntimeError("GROQ_API_KEY is missing. Add it to .env before running.")
		_client = Groq(api_key=api_key)

	return _client


def _convert_messages(messages: list) -> list[dict]:
	"""
	Convert LangChain messages to OpenAI-compatible role/content dicts.
	Consecutive same-role messages are merged for cleaner prompt windows.
	"""
	converted: list[dict] = []
	for msg in messages:
		if isinstance(msg, HumanMessage):
			role = "user"
		elif isinstance(msg, AIMessage):
			role = "assistant"
		else:
			continue

		content = msg.content if isinstance(msg.content, str) else str(msg.content)

		if converted and converted[-1]["role"] == role:
			converted[-1]["content"] += "\n" + content
		else:
			converted.append({"role": role, "content": content})

	return converted


def _chat_completion(
	messages: list[dict],
	*,
	system_prompt: str | None = None,
	max_tokens: int = 250,
	temperature: float = 0.2,
) -> str:
	"""Run a Groq chat completion and return assistant text."""
	payload_messages = messages.copy()
	if system_prompt:
		payload_messages = [{"role": "system", "content": system_prompt}] + payload_messages

	response = _get_client().chat.completions.create(
		model=MODEL_NAME,
		messages=payload_messages,
		temperature=temperature,
		max_tokens=max_tokens,
	)
	return (response.choices[0].message.content or "").strip()


def intent_node(state: AgentState) -> dict:
	"""Classify intent unless lead collection is in progress."""
	if state.get("awaiting_lead_field") is not None:
		return {"intent": "high_intent"}

	human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
	if not human_messages:
		return {"intent": "greeting"}

	last_message = human_messages[-1].content
	intent = classify_intent(last_message, _get_client(), MODEL_NAME)
	print(f"[INTENT] Classified as: {intent}")
	return {"intent": intent}


def rag_node(state: AgentState, retriever) -> dict:
	"""Answer product and policy questions using retrieved KB context."""
	human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
	last_query = human_messages[-1].content if human_messages else ""

	context = retrieve(last_query, retriever)
	print(f"[RAG] Retrieved context ({len(context)} chars)")

	system_prompt = f"""You are AutoStream's helpful sales assistant.
Answer the user's question using ONLY the retrieved context below.
If the answer is not in the context, reply: "That information is not available in the current knowledge base."
Do not invent or assume any facts beyond the provided context.
Be concise and friendly.
After answering, ask if they'd like to get started or have more questions.

CONTEXT:
{context}"""

	chat_messages = _convert_messages(state["messages"])
	ai_text = _chat_completion(
		chat_messages,
		system_prompt=system_prompt,
		max_tokens=400,
		temperature=0.2,
	)
	return {"messages": [AIMessage(content=ai_text)]}


def lead_collect_node(state: AgentState) -> dict:
	"""Collect lead data in strict order: name -> email -> platform."""
	updates: dict = {}

	human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
	last_input = human_messages[-1].content.strip() if human_messages else ""

	awaiting = state.get("awaiting_lead_field")

	if awaiting == "name":
		updates["lead_name"] = last_input
		awaiting = "email"
	elif awaiting == "email":
		if not re.fullmatch(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", last_input):
			updates["awaiting_lead_field"] = "email"
			updates["messages"] = [
				AIMessage(content="Please provide a valid email address.")
			]
			return updates

		updates["lead_email"] = last_input
		awaiting = "platform"
	elif awaiting == "platform":
		updates["lead_platform"] = last_input
		awaiting = "__ready__"

	name = updates.get("lead_name") or state.get("lead_name")

	if awaiting == "__ready__":
		updates["awaiting_lead_field"] = "__ready__"
		return updates

	if awaiting is None or awaiting == "name":
		response_text = (
			"That's exciting! I'd love to get you set up with AutoStream's Pro plan. "
			"To create your account, could I start with your full name?"
		)
		updates["awaiting_lead_field"] = "name"
	elif awaiting == "email":
		response_text = (
			f"Thanks, {name}! What's the best email address for your AutoStream account?"
		)
		updates["awaiting_lead_field"] = "email"
	elif awaiting == "platform":
		response_text = (
			"Perfect! And which platform do you primarily create content on? "
			"(for example: YouTube, Instagram, TikTok, Twitch)"
		)
		updates["awaiting_lead_field"] = "platform"
	else:
		response_text = "Let me get a couple of details to set up your account."

	updates["messages"] = [AIMessage(content=response_text)]
	return updates


def tool_node(state: AgentState) -> dict:
	"""Call mock lead capture exactly once after all lead fields are present."""
	name = (state.get("lead_name") or "").strip()
	email = (state.get("lead_email") or "").strip()
	platform = (state.get("lead_platform") or "").strip()

	if not all([name, email, platform]):
		print(
			"[TOOL] WARNING: Incomplete lead data - "
			f"name={name!r}, email={email!r}, platform={platform!r}"
		)
		return {
			"messages": [
				AIMessage(content="I'm missing a couple of details. Let me ask again.")
			],
			"awaiting_lead_field": None,
		}

	tool_result = mock_lead_capture(name, email, platform)
	print(f"[TOOL] Result: {tool_result}")

	closing_prompt = f"""The user {name} has just signed up for AutoStream's Pro plan.
Their email is {email} and they create content on {platform}.
Write a warm, enthusiastic 2-sentence closing message confirming their signup,
mentioning their platform, and telling them the team will reach out soon.
Keep it under 60 words. Be friendly and genuine."""

	closing_text = _chat_completion(
		[{"role": "user", "content": closing_prompt}],
		max_tokens=150,
		temperature=0.3,
	)

	return {
		"messages": [AIMessage(content=closing_text)],
		"lead_captured": True,
		"awaiting_lead_field": None,
	}


def general_node(state: AgentState) -> dict:
	"""Handle greetings, casual chat, and post-capture messages."""
	if state.get("lead_captured"):
		return {"messages": [AIMessage(content=POST_CAPTURE_SAFE_RESPONSE)]}

	system_prompt = """You are AutoStream's friendly AI assistant.
AutoStream is a SaaS platform that provides automated video editing tools for content creators.
Be warm, concise, and helpful. If the user seems interested in the product, invite them to ask
about pricing or features. Keep responses under 3 sentences."""

	chat_messages = _convert_messages(state["messages"])
	response_text = _chat_completion(
		chat_messages,
		system_prompt=system_prompt,
		max_tokens=200,
		temperature=0.3,
	)

	return {"messages": [AIMessage(content=response_text)]}
