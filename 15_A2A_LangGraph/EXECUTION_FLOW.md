# Execution Flow: `uv run python simple_agent.py`

This document explains step-by-step what happens when you run `simple_agent.py`.

**Note**: This document describes the flow of `simple_agent.py` (the client), NOT `app/__main__.py` (the A2A server). The `main()` function referenced here is from `simple_agent.py`, not from the server application.

---

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Script Entry Point (simple_agent.py __main__)          │
│    └─> asyncio.run(main())                                 │
│    Note: This is main() from simple_agent.py, not app/__main__.py│
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. main() Function (from simple_agent.py)                   │
│    ├─> Creates SimpleAgent instance                         │
│    ├─> Calls agent.initialize()                            │
│    └─> Calls agent.run_all_personas()                      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. For Each Persona (ml_expert, business_analyst, student)│
│    └─> agent.run_persona(persona_name)                     │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. LangGraph Execution Loop                                 │
│    └─> graph.astream(initial_state)                        │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Print Summary                                            │
│    └─> Display results for all 3 personas                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Step-by-Step Flow

### Phase 1: Initialization (Lines 546-509)

**Step 1.1: Script Execution**
```python
if __name__ == "__main__":
    asyncio.run(main())  # Entry point
```

**Step 1.2: Environment Setup**
- `load_dotenv()` loads environment variables from `.env` file
- Sets up logging with `logging.basicConfig(level=logging.INFO)`

**Step 1.3: Create SimpleAgent Instance**
```python
agent = SimpleAgent(max_queries_per_persona=3)
```
- Initializes:
  - `a2a_base_url = "http://localhost:10000"`
  - `max_queries_per_persona = 3`
  - `httpx_client = None` (will be created later)
  - `a2a_client = None` (will be created later)
  - `model = ChatOpenAI(...)` for persona reasoning

**Step 1.4: Initialize A2A Connection**
```python
await agent.initialize()
```
- Creates `httpx.AsyncClient` with 60s timeout
- Creates `A2ACardResolver` to fetch agent card from A2A server
- Fetches agent card from `http://localhost:10000/.well-known/agent-card`
- Creates `A2AClient` with the fetched agent card
- **Note**: The A2A server must be running (`uv run python -m app`)

---

### Phase 2: Run All Personas (Lines 476-491)

**Step 2.1: Loop Through Personas**
```python
for persona_name in PERSONAS.keys():
    # Runs: 'ml_expert', 'business_analyst', 'student'
```

**Step 2.2: For Each Persona**
- Calls `agent.run_persona(persona_name)` sequentially
- Each persona runs independently (one after another)

---

### Phase 3: Run Single Persona (Lines 425-474)

**Step 3.1: Setup Initial State**
```python
initial_state: PersonaState = {
    "messages": [],                    # Empty conversation
    "persona": persona_name,           # e.g., "ml_expert"
    "persona_config": PERSONAS[persona_name],  # Full persona config
    "a2a_response": "",                # No response yet
    "context_id": None,                # No context yet
    "task_id": None,                   # No task yet
    "query_count": 0,                  # Starting at 0
    "max_queries": 3,                  # Will stop after 3 queries
}
```

**Step 3.2: Build LangGraph**
```python
graph = self.build_persona_graph()
```
This creates a LangGraph with the following structure:

```
START
  │
  ▼
prepare_query  ──────>  call_a2a  ──────>  should_continue
  │                                              │
  │                                              │
  └──────────────────────────────────────────────┘
                    (if continue)
```

**Step 3.3: Execute Graph**
```python
async for state in graph.astream(initial_state):
    # Streams state updates as graph executes
```

---

### Phase 4: LangGraph Execution Loop

The graph executes in cycles. Each cycle consists of:

#### Cycle Iteration (Query #1, #2, #3)

**Node 1: `prepare_query` (Lines 291-330)**

**For First Query (query_count == 0):**
```python
query = persona_config["initial_query"]
# Uses the predefined initial query from PERSONAS dict
```

**For Follow-up Queries (query_count > 0):**
```python
# Builds conversation context from previous messages
conversation_context = "\n\nPrevious conversation:\n..."
# Uses LLM to generate follow-up question based on:
# - Persona's system_instruction
# - Previous conversation history
# - Persona's goals and personality
```

**Output:**
- Adds `HumanMessage(content=query)` to state.messages
- Returns updated state

---

**Node 2: `call_a2a` (Lines 332-356)**

**Step 4.2.1: Extract Query**
```python
last_message = state["messages"][-1]
query = last_message.content
```

**Step 4.2.2: Call A2A Agent**
```python
response_text, context_id, task_id = await self.call_a2a_agent(
    query, 
    state.get("context_id"),  # None for first query, set for follow-ups
    state.get("task_id")      # Not used (kept for compatibility)
)
```

**Inside `call_a2a_agent` (Lines 131-286):**

1. **Build Request Payload:**
   ```python
   send_message_payload = {
       "message": {
           "role": "user",
           "parts": [{"kind": "text", "text": query}],
           "message_id": uuid4().hex,
       },
   }
   ```

2. **Add Context (if follow-up):**
   ```python
   if context_id:
       send_message_payload["message"]["context_id"] = context_id
   # NOTE: task_id is NOT sent for follow-ups
   ```

3. **Send Request:**
   ```python
   request = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**send_message_payload))
   response = await self.a2a_client.send_message(request)
   ```

4. **Extract Response:**
   - Checks for errors first (JSONRPCErrorResponse)
   - Extracts text from nested structure: `root.result.artifacts[].parts[].root.text`
   - Falls back to regex if structure parsing fails
   - Extracts `context_id` from response for next query

5. **Return:**
   - `response_text`: The actual text response
   - `new_context_id`: For maintaining conversation context
   - `new_task_id`: Not used (set to None)

**Step 4.2.3: Update State**
```python
return {
    **state,
    "a2a_response": response_text,
    "context_id": context_id,        # Updated from A2A response
    "task_id": task_id,              # Not used
    "messages": state["messages"] + [AIMessage(content=response_text)],
    "query_count": state["query_count"] + 1,
}
```

---

**Node 3: `should_continue` (Lines 358-407)**

**Step 4.3.1: Check Hard Limits**
```python
if state["query_count"] >= state["max_queries"]:
    return "end"  # Stop after 3 queries
```

**Step 4.3.2: Check Response Quality**
```python
if not last_response or len(last_response.strip()) == 0:
    return "end"  # Empty response, stop
```

**Step 4.3.3: LLM Decision**
```python
# Uses LLM to decide if persona wants to continue
decision_prompt = f"""You are {persona_config['name']}.
...
Based on your persona and goals, do you want to ask a follow-up question?
Respond with 'YES' if you want to ask more, or 'NO' if you're satisfied.
"""
decision_response = self.model.invoke([("user", decision_prompt)])
decision = decision_response.content.strip().upper()
return "continue" if "YES" in decision else "end"
```

**Step 4.3.4: Conditional Edge**
- If `"continue"`: Go back to `prepare_query` (next iteration)
- If `"end"`: Go to `END` (finish persona)

---

### Phase 5: Complete Persona & Collect Results

**Step 5.1: Graph Completes**
- Final state is collected from the last `call_a2a` node

**Step 5.2: Return Results**
```python
return {
    "persona": persona_name,
    "persona_name": persona_config["name"],
    "messages": final_state.get("messages", []),  # All conversation messages
    "query_count": final_state.get("query_count", 0),
}
```

**Step 5.3: Move to Next Persona**
- Repeats Phase 3-5 for next persona

---

### Phase 6: Print Summary (Lines 514-538)

**Step 6.1: Display Header**
```
============================================================
SUMMARY
============================================================
```

**Step 6.2: For Each Persona Result**
```python
print(f"✅ {persona_name}")
print(f"   Queries made: {query_count}")
print(f"   Conversation messages: {len(messages)}")
print("\n   Conversation Preview:")
for i, msg in enumerate(messages, 1):
    role = "Persona" if HumanMessage else "A2A Agent"
    preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
    print(f"   {i}. [{role}]: {preview}")
```

**Step 6.3: Cleanup**
```python
await agent.close()  # Closes httpx client
```

---

## Complete Flow Example (One Persona)

```
┌─────────────────────────────────────────────────────────────┐
│ Persona: ml_expert                                          │
│ Initial Query: "What are the latest developments..."        │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Cycle 1: Query #1                                           │
│ ├─> prepare_query: Uses initial_query                       │
│ ├─> call_a2a: Sends to A2A agent (no context_id)           │
│ │   └─> A2A agent processes with tools (Tavily, ArXiv, RAG)│
│ │   └─> Returns response + context_id                       │
│ └─> should_continue: LLM decides "YES" → continue          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Cycle 2: Query #2                                           │
│ ├─> prepare_query: LLM generates follow-up based on        │
│ │                  persona + conversation history           │
│ ├─> call_a2a: Sends with context_id (maintains context)     │
│ │   └─> A2A agent processes with conversation context        │
│ │   └─> Returns response + updated context_id               │
│ └─> should_continue: LLM decides "YES" → continue          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Cycle 3: Query #3                                           │
│ ├─> prepare_query: LLM generates final follow-up            │
│ ├─> call_a2a: Sends with context_id                          │
│ │   └─> A2A agent processes                                  │
│ │   └─> Returns response                                    │
│ └─> should_continue: query_count >= 3 → "end"              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Complete: Return results with all messages                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### 1. **State Management**
- `PersonaState` is a TypedDict that holds all conversation state
- LangGraph automatically manages state updates between nodes
- `messages` uses `Annotated[List, add_messages]` for proper message merging

### 2. **Context ID vs Task ID**
- **First Query**: No `context_id` or `task_id` → starts new conversation
- **Follow-up Queries**: Only `context_id` is used → maintains conversation context
- **Why**: `task_id` becomes invalid after task completes. Reusing it causes errors.

### 3. **Persona Behavior**
- Each persona has:
  - `system_instruction`: Defines personality and goals
  - `initial_query`: First question to ask
  - LLM-generated follow-ups based on conversation history

### 4. **A2A Protocol**
- Uses `A2AClient` to communicate with main agent
- Sends JSON-RPC 2.0 requests
- Receives structured responses with artifacts
- Extracts text from nested response structure

### 5. **LangGraph Flow**
- **START** → `prepare_query` → `call_a2a` → `should_continue`
- If continue: `should_continue` → `prepare_query` (loop)
- If end: `should_continue` → **END**

---

## Prerequisites

Before running `simple_agent.py`, ensure:

1. **A2A Server is Running:**
   ```bash
   uv run python -m app
   ```
   Server should be listening on `http://localhost:10000`

2. **Environment Variables Set:**
   - `OPENAI_API_KEY`: For LLM calls
   - `TOOL_LLM_NAME`: Model name (default: "gpt-4o-mini")
   - `TOOL_LLM_URL`: API URL (default: "https://api.openai.com/v1")
   - `TAVILY_API_KEY`: For web search tool

3. **Dependencies Installed:**
   ```bash
   uv sync
   ```

---

## Expected Output

```
🚀 Simple Agent with 3 Personas
============================================================

This agent will demonstrate 3 different personas interacting
with the A2A agent through LangGraph.

[INFO] Fetching agent card from http://localhost:10000
[INFO] Successfully fetched agent card
[INFO] A2A client initialized

============================================================
Starting conversation with Dr. Sarah Chen - ML Expert
============================================================

[INFO] [ml_expert] Querying A2A agent: What are the latest developments...
[INFO] [ml_expert] Received response: Based on recent research...

[INFO] [ml_expert] Querying A2A agent: Can you provide specific...
[INFO] [ml_expert] Received response: Here are some key papers...

[INFO] [ml_expert] Querying A2A agent: What about the architecture...
[INFO] [ml_expert] Received response: The architecture involves...

============================================================
Conversation with Dr. Sarah Chen - ML Expert completed
Total queries: 3
============================================================

[... Similar output for business_analyst and student ...]

============================================================
SUMMARY
============================================================
✅ Dr. Sarah Chen - ML Expert
   Queries made: 3
   Conversation messages: 6

   Conversation Preview:
   1. [Persona]: What are the latest developments in transformer...
   2. [A2A Agent]: Based on recent research...
   3. [Persona]: Can you provide specific academic papers...
   4. [A2A Agent]: Here are some key papers...
   5. [Persona]: What about the architecture details...
   6. [A2A Agent]: The architecture involves...

[... Similar summaries for other personas ...]
```

---

## Troubleshooting

### Error: "Failed to initialize A2A client"
- **Cause**: A2A server not running
- **Fix**: Start server with `uv run python -m app`

### Error: "API key error detected"
- **Cause**: Missing or invalid `OPENAI_API_KEY`
- **Fix**: Check `.env` file has valid API key

### Error: "Task is in terminal state: completed"
- **Cause**: Trying to reuse `task_id` (shouldn't happen with current code)
- **Fix**: Code already fixed - only uses `context_id` for follow-ups

### Error: "Unable to extract response text"
- **Cause**: A2A response structure changed or unexpected format
- **Fix**: Check A2A server logs, verify response structure

