# Research Team Graph Flow Diagram

## Mermaid Diagram

```mermaid
graph TD
    START([START]) --> ResearchSupervisor[ResearchSupervisor<br/>LLM Router<br/>Entry Point]
    
    ResearchSupervisor -->|Decision: next = "HowPeopleUseAIRetriever"| HowPeopleUseAIRetriever[HowPeopleUseAIRetriever<br/>RAG Agent<br/>- Uses retrieve_information tool<br/>- Searches vector store<br/>- Returns AI usage data]
    
    ResearchSupervisor -->|Decision: next = "Search"| Search[Search<br/>Tavily Agent<br/>- Uses tavily_tool<br/>- Searches web for up-to-date info]
    
    HowPeopleUseAIRetriever -->|Always returns to supervisor| ResearchSupervisor
    Search -->|Always returns to supervisor| ResearchSupervisor
    
    ResearchSupervisor -->|Decision: next = "FINISH"| END([END])
    
    classDef supervisor fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef agent fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef terminal fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    
    class ResearchSupervisor supervisor
    class HowPeopleUseAIRetriever,Search agent
    class START,END terminal
```

## Alternative Text-Based Diagram

```
                    START
                      │
                      ▼
            ┌─────────────────────┐
            │ ResearchSupervisor  │ ◄─── Entry Point
            │   (LLM Router)      │
            └─────────────────────┘
                      │
                      │ (Decision: next = "HowPeopleUseAIRetriever")
                      ▼
            ┌─────────────────────┐
            │ HowPeopleUseAIRetriever │
            │   (RAG Agent)       │
            │   - Uses retrieve_information tool │
            │   - Searches vector store │
            │   - Returns AI usage data │
            └─────────────────────┘
                      │
                      │ (Always returns to supervisor)
                      ▼
            ┌─────────────────────┐
            │ ResearchSupervisor  │
            │   (LLM Router)      │
            └─────────────────────┘
                      │
                      │ (Decision: next = "FINISH")
                      ▼
                     END
```

## Key Components:

1. **ResearchSupervisor**: 
   - Entry point of the graph
   - LLM-based router that decides which agent acts next
   - Can route to "Search", "HowPeopleUseAIRetriever", or "FINISH"

2. **HowPeopleUseAIRetriever**: 
   - RAG agent that searches the vector store
   - Uses the `retrieve_information` tool
   - Provides specific information about AI usage from the PDF

3. **Search**: 
   - Web search agent using Tavily
   - Uses the `tavily_tool` for up-to-date information
   - Can find current information not in the vector store

4. **Flow Control**:
   - All agent nodes return to the supervisor after completing their task
   - Supervisor makes intelligent decisions about next steps
   - Graph ends when supervisor decides "FINISH"

## State Flow:
- **messages**: Accumulates conversation history using `operator.add`
- **team_members**: List of available agents ["Search", "HowPeopleUseAIRetriever"]
- **next**: Current routing decision made by supervisor
