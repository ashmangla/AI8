
"""LangGraph agent integration with production features."""

from typing import Dict, Any, List, Optional
import os

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from .models import get_openai_model
from .rag import ProductionRAGChain
from .guardrails import (
    create_guardrails_guard,
    create_guardrails_node,
    create_guardrails_node_async,
)
from guardrails import Guard


class AgentState(TypedDict):
    """State schema for agent graphs."""

    messages: Annotated[List[BaseMessage], add_messages]
    validation_results: Optional[Dict[str, Any]]


def create_rag_tool(rag_chain: ProductionRAGChain):
    """Create a RAG tool from a ProductionRAGChain."""

    @tool
    def retrieve_information(query: str) -> str:
        """Use Retrieval Augmented Generation to retrieve information from the student loan documents."""
        try:
            result = rag_chain.invoke(query)
            return result.content if hasattr(result, "content") else str(result)
        except Exception as e:  # pragma: no cover - logging/observability layer would handle in prod
            return f"Error retrieving information: {str(e)}"

    return retrieve_information


def get_default_tools(rag_chain: Optional[ProductionRAGChain] = None) -> List:
    """Get default tools for the agent."""

    tools = []

    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=5))

    # Add Arxiv tool
    tools.append(ArxivQueryRun())

    # Add RAG tool if provided
    if rag_chain:
        tools.append(create_rag_tool(rag_chain))

    return tools


def create_langgraph_helpfulness_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    input_guard: Optional[Guard] = None,
    output_guard: Optional[Guard] = None,
    async_guard_validation: bool = False,
):
    """Create a LangGraph agent with a post-response helpfulness check."""

    if tools is None:
        tools = get_default_tools(rag_chain)

    # Create default guards if not provided
    if input_guard is None:
        input_guard = create_guardrails_guard(
            enable_jailbreak_detection=True,
            enable_pii_protection=True,
            enable_profanity_check=True,
        )

    if output_guard is None:
        output_guard = create_guardrails_guard(
            enable_jailbreak_detection=False,
            enable_pii_protection=True,
            enable_profanity_check=True,
        )

    guard_node_factory = (
        create_guardrails_node_async if async_guard_validation else create_guardrails_node
    )

    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)

    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""

        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def route_to_action_or_helpfulness(state: AgentState):
        """Decide whether to execute tools or run the helpfulness evaluator."""

        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return "helpfulness"

    def helpfulness_node(state: AgentState) -> Dict[str, Any]:
        """Evaluate helpfulness of the latest response relative to the initial query."""

        existing_results = state.get("validation_results")
        if isinstance(existing_results, dict):
            updated_results = dict(existing_results)
            guardrails_results = list(updated_results.get("guardrails", []))
        elif isinstance(existing_results, list):
            guardrails_results = list(existing_results)
            updated_results = {}
        else:
            guardrails_results = []
            updated_results = {}

        if guardrails_results:
            updated_results["guardrails"] = guardrails_results

        # If we've exceeded loop limit, short-circuit with END decision marker
        if len(state["messages"]) > 10:
            updated_results["helpfulness_decision"] = "END"
            return {"validation_results": updated_results}

        initial_query = state["messages"][0]
        final_response = state["messages"][-1]

        prompt_template = """
    Given an initial query and a final response, determine if the final response is extremely helpful or not. Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.

    Initial Query:
    {initial_query}

    Final Response:
    {final_response}"""

        helpfulness_prompt_template = PromptTemplate.from_template(prompt_template)
        helpfulness_check_model = get_openai_model(model_name="gpt-4.1-mini", temperature=temperature)
        helpfulness_chain = (
            helpfulness_prompt_template | helpfulness_check_model | StrOutputParser()
        )

        helpfulness_response = helpfulness_chain.invoke(
            {
                "initial_query": initial_query.content,
                "final_response": final_response.content,
            }
        )

        decision = "Y" if "Y" in helpfulness_response else "N"
        updated_results["helpfulness_decision"] = decision
        return {"validation_results": updated_results}

    def helpfulness_decision(state: AgentState):
        """Terminate on 'HELPFULNESS:Y' or loop otherwise; guard against infinite loops."""

        results = state.get("validation_results") or {}
        decision = results.get("helpfulness_decision") if isinstance(results, dict) else None

        if decision == "END":
            return END
        if decision == "Y":
            return "end"
        return "continue"

    def should_continue(state: AgentState):
        """Route to tools if the last message has tool calls."""

        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return "output_guard"

    # Create guard nodes
    input_guard_node = guard_node_factory(
        input_guard=input_guard,
        output_guard=None,
        strict_mode=True,
    )

    output_guard_node = guard_node_factory(
        input_guard=None,
        output_guard=output_guard,
        strict_mode=True,
    )

    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)

    # Add nodes
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("helpfulness", helpfulness_node)
    graph.add_node("output_guard", output_guard_node)

    # Set entry point to input_guard
    graph.set_entry_point("input_guard")

    # Flow: input_guard -> agent
    graph.add_edge("input_guard", "agent")

    # Conditional from agent: tool calls -> action, otherwise -> helpfulness
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_helpfulness,
        {"action": "action", "helpfulness": "helpfulness"},
    )
    graph.add_conditional_edges(
        "helpfulness",
        helpfulness_decision,
        {"continue": "agent", "end": "output_guard"},
    )

    # Action loops back to agent
    graph.add_edge("action", "agent")

    # Output guard leads to END
    graph.add_edge("output_guard", END)

    return graph.compile()


graph = create_langgraph_helpfulness_agent()
