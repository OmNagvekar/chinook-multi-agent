## Chinook Multi‑Agent Music Store README

This repository demonstrates a multi‑agent system built with LangGraph Supervisor, orchestrating three specialized agents over the Chinook sample database:

* **CatalogAgent** – Browse tracks by genre
* **OrderAgent** – Create invoices (orders)
* **ChinookDBAgent** – Run arbitrary safe SQL SELECT queries

A **MusicStoreSupervisor** agent delegates user requests to the right worker agent, collects their outputs, and then composes a final, consolidated answer.

---

### Multi-Agent Graph Architecture Image

![alt text](https://github.com/OmNagvekar/chinook-multi-agent/blob/main/Multi_agent_architecture.png?raw=true)

---

### 📁 Repository Structure

```
.
├── README.md
├── ChinHook_database.db       # SQLite Chinook sample DB
└── agents.py      # Agent & supervisor definitions
```

---

### 🔧 Prerequisites

* Python 3.11+
* [pipenv](https://pipenv.pypa.io/) or [venv](https://docs.python.org/3/library/venv.html)
* A valid `.env` file with your Google or OpenAI API credentials, e.g.:

  ```.env
  GOOGLE_API_KEY=
  ```

---

### ⚙️ Installation

```bash
# 1. Clone repo
git clone https://github.com/OmNagvekar/chinook-multi-agent.git
cd chinook-multi-agent

# 2. Create virtualenv & install deps
pip install -r requirements.txt

# 3. Verify Chinook DB is present
ls ChinHook_database.db
```

---

### 🛠️ Configuration

Make sure your `.env` includes API keys for whichever LLM backend you’re using:

```bash
# Example .env
GOOGLE_API_KEY=ya29-…
```

---

### 🚀 Running the Demo

In `agents.py`, the `__main__` block demonstrates two interactions:

```bash
python agents.py
```

1. **Browse Rock tracks**
2. **(Optional)** Continue the same session to “buy” tracks

The supervisor prints each step via `pretty_print()`.

---

### 🔍 Example Prompts

Use these sample user inputs to test functionality:

1. **Catalog browsing**

   ```
   Show me 3 Rock tracks available in the store
   ```

2. **Custom SQL query**

   ```
   How many invoices does customer 12 have?
   ```

3. **Place an order**

   ```
   I'd like to checkout: Customer 12, cart = [{'TrackId': 1, 'UnitPrice': 0.99, 'Quantity': 2}]
   ```

4. **Combined flow**

   ```
   First show me Jazz tracks, then I'll buy one.
   ```

---

### 🧩 How It Works

1. **Tools** (`@tool` functions) interact with the SQLite Chinook DB:

   * `catalog_search(genre)`
   * `database_search(query)`
   * `create_order(customer_id, cart)`

2. **Agents** wrap these tools with system prompts via `create_react_agent`:

   * `CatalogAgent`
   * `OrderAgent`
   * `ChinookDBAgent`

3. **Supervisor** (`create_supervisor`) uses handoff tools to route each user query to exactly one worker agent. Upon receiving an agent’s response, the supervisor:

   * Summarizes all agents’ outputs so far
   * Crafts one final, user‑facing answer

4. **State & Memory** are managed with `InMemorySaver`.

---

### 🧪 Testing & Validation

Run the provided test harness in `music_store_agents.py` or integrate your own:

```python
from music_store_agents import supervisor
from langchain_core.messages import HumanMessage

resp = supervisor.invoke({
    "messages": [{"role":"user","content":"List pop tracks"}]
})
print(resp["messages"][-1].content)
```

Check for correct delegation, tool execution, and coherent final summary.

---

### 📈 Extensibility

* **Add Agents**: e.g., `RecommendationAgent`, `AnalyticsAgent`
* **Swap LLMs**: configure `ChatGoogleGenerativeAI` or `init_chat_model("openai:gpt-4")`
* **Persistent Memory**: replace `InMemorySaver` with a database‑backed saver

---

### 📜 License

This code is released under the MIT License. See [LICENSE](LICENSE) for details.
