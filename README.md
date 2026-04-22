# AutoStream AI Agent

A conversational AI agent for AutoStream (a fictional SaaS) built with LangGraph, Groq, and FAISS RAG. It answers product questions from a local knowledge base and converts high-intent conversations into qualified leads.

## Features

- Intent classification
- RAG-powered product Q&A
- High-intent lead capture
- Email validation before tool execution
- Stateful multi-turn conversation with LangGraph

## Quick Start

1. Clone the repo and enter the folder.
2. Create and activate a virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Copy `.env.example` to `.env` and set `GROQ_API_KEY`.
5. Run `python main.py`.

## Architecture

This project uses LangGraph because it provides an explicit and inspectable state machine for multi-turn conversation flows. The state is defined in a typed `AgentState` dictionary and persists across turns, including full message history, current intent, lead fields, and lead-capture status. The graph enters through an intent node and conditionally routes to either a general assistant node, a RAG response node, or a lead collection node.

The RAG pipeline is local-first. At startup, the app loads `knowledge_base/autostream_kb.md`, splits it into chunks, embeds those chunks with `all-MiniLM-L6-v2`, and builds a FAISS index once. Query-time retrieval returns the top two chunks and grounds the assistant response to reduce hallucinations.

Intent classification is performed by a Groq-hosted model as a strict JSON classifier with three labels: `greeting`, `inquiry`, and `high_intent`. When high intent is detected, a sub-state flow collects lead details in fixed order: name, then email, then platform. Email input is validated before progressing. A guard prevents tool execution until all fields are present, then `mock_lead_capture` is called exactly once.

## WhatsApp Deployment

The current implementation is CLI-first for easy local validation. To deploy on WhatsApp, keep the same graph and move transport to a webhook service:

1. Use Meta WhatsApp Cloud API webhook callbacks for inbound messages.
2. Store per-user state externally (Redis or database) keyed by phone number.
3. On each incoming message, append to that user's state and invoke the graph.
4. Send the resulting assistant message back with the WhatsApp send-message API.
5. Keep lead capture as a separate backend endpoint and replace the mock tool with a real POST call.

## Project Structure

```text
ServiceHiveV1/
|-- agent/
|   |-- __init__.py
|   |-- graph.py
|   |-- intent.py
|   |-- nodes.py
|   |-- state.py
|   `-- tools.py
|-- knowledge_base/
|   `-- autostream_kb.md
|-- rag/
|   |-- __init__.py
|   |-- loader.py
|   `-- retriever.py
|-- main.py
|-- requirements.txt
|-- .env.example
|-- .gitignore
`-- README.md
```

## Demo Video

<video src="AutoStream_AI_Agent_Demo_Anand_Raj.mp4" controls width="800"></video>