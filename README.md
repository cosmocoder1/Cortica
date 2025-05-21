# 🧠 Cortica

**Cortica** is a lightweight AGI-aligned memory module that gives stateless LLMs the illusion of continuity.
It stores user messages as embedded vectors and retrieves the most relevant ones to build context-aware prompts.

Use it to give *semantic memory* to chatbots, agents, or RAG systems — no tagging, NLP, or databases required.

---

## 🚀 What Cortica Does

* Stores user messages with vector embeddings
* Recalls the most relevant past messages for a new query
* Builds clean, token-aware context blocks to inject into LLM prompts
* Keeps your agent **on track**, even across drifting conversations

---

## 🧠 Ideal Use Case

> Cortica is the **working memory** of your LLM app.
> It doesn't replace vector DBs like Chroma — it complements them by handling **what the user just said**, not what your corpora said last year.

Use it for:

* Customer support bots that remember the user's last 5 complaints
* Product recommendation agents that adapt to feedback across turns
* Lightweight RAG systems that need local, per-user awareness

---

🔌 Example: Customer Support Chatbot

```
from cortica.cortex import Cortex
from langchain_openai import OpenAIEmbeddings

openai_key = os.getenv("OPENAI_API_KEY")

embedder = OpenAIEmbeddings(openai_api_key=openai_key)
cortex = Cortex(embedder=embedder)


# Store previous customer messages

cortex.remember("My order arrived damaged.")
cortex.remember("I already submitted a return request yesterday.")


# Build context for the assistant to use in its next LLM call

context = cortex.build_context_prompt("Can you check the status of my return?")

Example output:

### Prior User Context
The following entries summarize what the user has previously communicated.
Use this context to respond in an informed, user-specific fashion.

- My order arrived damaged.
- I already submitted a return request yesterday.

```
---

## 🔧 Architecture

* `Cortex`: The high-level memory API
* `MemoryGraph`: Vector store for embedding-based recall
* User-supplied embedder with:

```python
def embed_query(text: str) -> List[float]
```

No tagging.
No NLP pipelines.
No database dependencies.

---

## ✨ Features

* **Memory Storage** — Store user utterances and metadata
* **Semantic Recall** — Retrieve past context via cosine similarity
* **LLM-Ready Prompts** — Structured context blocks with token trimming
* **Token Budgeting** — Avoid prompt overflow with dynamic limits
* **Plug & Play** — Works with OpenAI, HuggingFace, or custom embedders

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/Cortica.git
cd Cortica
pip install -e .
```

> Requires Python 3.10+

---

## 🔮 Coming Soon

* Disk-based memory persistence
* Memory visualization + session replay
* LangChain + Streamlit integrations
* Concept summarization + compression modules

---

## 📄 License

MIT License
Cortica is free to use, fork, and integrate into your intelligent systems.

---

Made with clarity and memory in mind.
