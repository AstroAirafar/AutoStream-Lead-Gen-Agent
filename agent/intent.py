import json
import re

from groq import Groq


INTENT_SYSTEM_PROMPT = """You are an intent classifier for AutoStream, a SaaS video editing platform.

Given the user's latest message, classify it into EXACTLY ONE of these intents:
- "greeting": General hello, small talk, or messages with no specific product intent
- "inquiry": Questions about features, pricing, plans, policies, or how the product works
- "high_intent": User expresses desire to sign up, try, purchase, subscribe, or start using the product

Respond with ONLY a raw JSON object and nothing else. No explanation. No markdown. No backticks.
Example response: {"intent": "inquiry"}"""


def _extract_json(text: str) -> dict | None:
	stripped = text.strip()

	if stripped.startswith("```"):
		stripped = stripped.strip("`")
		stripped = stripped.replace("json", "", 1).strip()

	try:
		return json.loads(stripped)
	except json.JSONDecodeError:
		match = re.search(r"\{[\s\S]*\}", text)
		if not match:
			return None
		try:
			return json.loads(match.group(0))
		except json.JSONDecodeError:
			return None


def classify_intent(last_user_message: str, client: Groq, model: str) -> str:
	"""Classify the latest user message with Groq and return a safe intent value."""
	try:
		response = client.chat.completions.create(
			model=model,
			temperature=0,
			max_tokens=60,
			messages=[
				{"role": "system", "content": INTENT_SYSTEM_PROMPT},
				{"role": "user", "content": last_user_message},
			],
		)
		raw = (response.choices[0].message.content or "").strip()
		parsed = _extract_json(raw) or {}
		intent = parsed.get("intent", "inquiry")

		if intent not in ("greeting", "inquiry", "high_intent"):
			return "inquiry"
		return intent
	except Exception:
		return "inquiry"
