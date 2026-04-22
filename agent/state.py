from typing import Annotated, List, Optional, TypedDict
import operator

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
	# Full conversation history across turns.
	messages: Annotated[List[BaseMessage], operator.add]

	# Intent for the latest user message.
	intent: str

	# Lead fields populated sequentially.
	lead_name: Optional[str]
	lead_email: Optional[str]
	lead_platform: Optional[str]

	# True after mock lead capture executes successfully.
	lead_captured: bool

	# Which lead field we are currently collecting.
	awaiting_lead_field: Optional[str]
