# Smart Financial Portfolio Analyst: Autonomous Compliance Audit Engine

A local, data-private **Cognitive Audit Engine** designed to evaluate unstructured portfolio rebalancing requests against institutional investment risk constraints in real-time. Built entirely using an air-gapped, state-driven multi-node orchestration loop, this system ensures that sensitive wealth management strategies and transactional intent are evaluated securely without leaking data to public cloud APIs.

## 🏗️ System Architecture

The core engineering pattern decouples data extraction, rule retrieval, and cognitive validation into a deterministic state-machine graph topology.


```

[User Input Query]
│
▼
┌───────────┐      ┌─────────────────────────┐
│ Analyzer  │ ───> │   Extracts Ticker       │
└───────────┘      └─────────────────────────┘
│
▼
┌───────────┐      ┌─────────────────────────┐
│  Tools    │ ───> │ Scrapes Live Market Data│ ◄─── [yfinance API]
└───────────┘      │ Queries Local Vector DB │ ◄─── [FAISS (Rules Vector Database)]
│              └─────────────────────────┘
▼
┌───────────┐      ┌─────────────────────────┐
│ Evaluator │ ───> │ Mathematical Validation │ ───> [Final Report Card Dashboard]
└───────────┘      └─────────────────────────┘

```

1. **Orchestration Layer (LangGraph):** Manages cyclic execution states across independent compute nodes (`analyzer` ➔ `tools` ➔ `evaluator`) to enforce architectural separation of concerns.
2. **Contextual Retrieval Layer (RAG):** Utilizes local document chunking embeddings (`all-MiniLM-L6-v2`) mapped inside a high-density **FAISS** vector instance to isolate compliance rules locally.
3. **Inference Execution:** Driven by a localized **Llama 3** engine to execute semantic intent extraction and reasoning boundaries completely offline.

---

## 🛠️ Tech Stack & Dependencies

* **Language Core:** Python 3.10
* **Agentic Framework:** LangGraph, LangChain Core / Community
* **Local Inference Engine:** Ollama (Llama 3 8B)
* **Vector Database Engine:** FAISS (Facebook AI Similarity Search)
* **Embedding Pipeline:** HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Data Scraper Layer:** yfinance API & Pandas Time-Series Filtering
* **Frontend UI Presentation:** Streamlit Web Dashboard Engine

---

## ⚡ Engineering Highlights & Problem Solving

### 🛑 Resolving Weekend Market Data Drops (Zero-Value Bug)
* **The Challenge:** During off-market intervals (weekends/holidays), standard financial scrapers like `yfinance` inject null records or zero-padded rows into historical dataframes. Blindly reading the latest array element (`iloc[-1]`) causes downstream UI components to crash or incorrectly display metrics as `$0.00`.
* **The Engineering Solution:** Implemented a robust data validation fallback layer using localized Pandas vector filtering. The script aggressively isolates records where `Close > 0.1` and steps backward chronologically to capture the absolute nearest active trading close (e.g., Friday's market session), ensuring system operational resilience 24/7.

### 🔒 Privacy-First Design
* By leveraging local embeddings and a local instance of Llama 3 via Ollama, this project complies with strict institutional data-handling policies, proving that automated financial validation can exist without public network dependencies.

---
## 🛡️ Governance & Guardrail Architecture

To ensure institutional safety, the system enforces strict runtime boundaries across three structural dimensions:

1. **Input Validation Guardrails (Topic Filtering):**
   * Before reaching the cognitive routing loop, user inputs are structurally sanitized. If a query contains injection vectors, malformed logic, or requests un-vetted asset classes (such as unregulated crypto or speculative micro-caps), the engine short-circuits execution. It flags a policy violation without invoking the vector DB or external market APIs, completely preventing prompt injection exploits.

2. **Retrieval-Augmented Generation (RAG) Guardrails:**
   * The system relies exclusively on context fetched from the local FAISS index (your institutional compliance rules). 
   * The system prompt is engineered with strict constraint bounding instructions: *"Base your evaluation strictly on the retrieved context guidelines. If the guidelines do not contain rules pertaining to the ticker or allocation requested, default to an absolute REJECTION and do not extrapolate or guess."* This mitigates semantic drift and eliminates hallucinated guidelines.

3. **Output Structuring Guardrails (Deterministic Evaluation):**
   * The `evaluator` node does not output raw, open-ended conversational prose. It maps outputs directly into a typed schema (using Pydantic models/LangChain structured outputs). 
   * This forces the model to strictly isolate parameters into fields like `verdict` (`APPROVED` or `REJECTED`), `breached_policy_id`, and `mathematical_delta`. This structure guarantees that your Streamlit frontend receives stable data points to build its UI report cards without parsing unpredictable text.
-----

## 🚀 Local Installation & Setup

### Prerequisites
* Python 3.10 installed on your system.
* Ollama installed and running (`ollama run llama3`).

### Step-by-Step Execution

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/SNEHRANJAN28/smart-portfolio-agent.git](https://github.com/SNEHRANJAN28/smart-portfolio-agent.git)
   cd smart-portfolio-agent

```

2. **Initialize a Virtual Environment & Install Packages:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

```


3. **Construct the Compliance Vector DB:**
Ensure your compliance rules text files are positioned inside the root folder, then compile the dense matrix index:
```bash
python data_store.py

```


4. **Launch the Streamlit Web Application:**
```bash
streamlit run app.py

```



---

## 📊 Sample Test Scenarios

### Case A: Compliant Request (Within Thresholds)

* **Input:** *"I want to rebalance my portfolio and buy Microsoft stock. Please allocate 9% of my total capital directly into MSFT."*
* **System Action:** Graph maps ticker to `MSFT`, queries FAISS context (detecting a 10% maximum equity limit rule), extracts Friday's close data, confirms math logic ($9\% \le 10\%$), and returns a `✅ APPROVED` verdict report card.

### Case B: Non-Compliant Request (Breaching Thresholds)

* **Input:** *"Move 15% of my asset reserves into Microsoft stock immediately."*
* **System Action:** Engine identifies a threshold exception breach ($15\% > 10\%$) and dynamically generates an automated `❌ REJECTED` report with compliance context details.

```
