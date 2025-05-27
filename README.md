# 🧠 Cortica

**Cortica** is a lightweight AGI-aligned memory module that gives stateless LLMs the illusion of continuity.
It stores user messages as embedded vectors and retrieves the most relevant ones to build context-aware prompts.

Use it to give *semantic memory* to chatbots, agents, or RAG systems — no tagging, NLP, or databases required.

---

## 🚀 What Cortica Does

- Stores user messages with vector embeddings  
- Recalls the most relevant past messages for a new query  
- Builds clean, token-aware context blocks to inject into LLM prompts  
- Tracks **conversational tone over time** to help LLMs respond more intuitively  
- Keeps your agent **on track**, even across drifting conversations  

---

## ✨ Features

- **Memory Storage** — Store user utterances and metadata  
- **Semantic Recall** — Retrieve past context via cosine similarity  
- **LLM-Ready Prompts** — Structured context blocks with token trimming  
- **Tone Tracking** — Capture average valence/arousal over recent messages  
- **Token Budgeting** — Avoid prompt overflow with dynamic limits  
- **Plug & Play** — Works with OpenAI, HuggingFace, or custom embedders  

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
```
Example output:

```
You're speaking with a user.

### Prior User Context
These are the latest communications they've sent you.
Use this context to respond in an informed, user-specific fashion.

- I already submitted a return request yesterday. (1 min ago)
- I'm hoping to get a replacement quickly. (1 min ago)
- I ordered a coffee machine but it arrived damaged. (1 min ago)
- I love how fast your shipping usually is though! (1 min ago)
- My name is Sarah and I live in Chicago. (1 min ago)

### Conversational Tone Summary (across time)
These represent average emotional tone trends across three time horizons:
- **Last 2 messages**: valence=+0.50, arousal=+0.50
- **Last 10 messages**: valence=+0.20, arousal=+0.20
- **Overall**: valence=+0.20, arousal=+0.20
Use these to understand how the user's tone may be shifting over time so that you can be more adaptive in your communication.
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

## ⚙️ Installation

```bash
git clone https://github.com/cosmocoder1/Cortica.git
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
