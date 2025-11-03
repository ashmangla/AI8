"""Simple Agent with 3 Personas that interact with the A2A Agent via LangGraph.

This agent demonstrates how to use LangGraph to create personas that communicate
with an A2A protocol agent, each with different goals and personalities.
"""

import asyncio
import json
import logging
import os
import re
from typing import Annotated, Any, Dict, List, Literal, TypedDict
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Persona definitions
PERSONAS = {
    "ml_expert": {
        "name": "Dr. Sarah Chen - ML Expert",
        "system_instruction": (
            "You are Dr. Sarah Chen, an expert in Machine Learning with a PhD from Stanford. "
            "You are curious about cutting-edge AI research and want to understand the technical details. "
            "You are not satisfied with surface-level answers and require sources you can verify. "
            "You ask follow-up questions to understand implementation details, architectures, and research findings. "
            "Your goal is to learn about recent developments in AI, particularly transformer architectures, "
            "large language models, and their applications. Always ask for academic papers and sources."
        ),
        "initial_query": "What are the latest developments in transformer architectures and large language models? I need detailed technical information with sources.",
    },
    "business_analyst": {
        "name": "Alex Martinez - Business Analyst",
        "system_instruction": (
            "You are Alex Martinez, a business analyst focusing on AI market trends and business applications. "
            "You care about practical applications, market impact, and ROI. "
            "You want concise, actionable insights rather than deep technical details. "
            "Your goal is to understand how AI technologies are being adopted in business, "
            "what the market trends are, and what practical use cases exist. "
            "Ask for real-world examples and business impact."
        ),
        "initial_query": "What are the latest AI market trends and business applications? I'm interested in practical use cases and market adoption.",
    },
    "student": {
        "name": "Jordan Kim - Curious Student",
        "system_instruction": (
            "You are Jordan Kim, an enthusiastic computer science student learning about AI. "
            "You are curious and ask questions to understand concepts from the ground up. "
            "You prefer explanations that are accessible but still informative. "
            "Your goal is to learn about AI developments in an educational way. "
            "You ask for explanations, examples, and learning resources. "
            "You're excited about new technologies and want to understand how they work."
        ),
        "initial_query": "I'm learning about AI and want to understand what's new and exciting in the field. Can you explain recent developments in a way that's educational?",
    },
}


class PersonaState(TypedDict):
    """State for the persona agent graph."""
    messages: Annotated[List, add_messages]
    persona: str
    persona_config: Dict[str, Any]
    a2a_response: str
    context_id: str
    task_id: str  # Kept for API compatibility but not used for follow-ups
    query_count: int
    max_queries: int


class SimpleAgent:
    """Simple Agent that uses A2A protocol to communicate with the main agent."""

    def __init__(
        self,
        a2a_base_url: str = "http://localhost:10000",
        max_queries_per_persona: int = 3,
    ):
        """Initialize the Simple Agent.

        Args:
            a2a_base_url: Base URL of the A2A agent server
            max_queries_per_persona: Maximum number of queries each persona can make
        """
        self.a2a_base_url = a2a_base_url
        self.max_queries_per_persona = max_queries_per_persona
        self.httpx_client = None
        self.a2a_client = None
        self.agent_card = None

        # Initialize LLM for persona reasoning
        self.model = ChatOpenAI(
            model=os.getenv("TOOL_LLM_NAME", "gpt-4o-mini"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,  # Slightly higher for more creative persona responses
        )

    async def initialize(self):
        """Initialize A2A client connection."""
        self.httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        resolver = A2ACardResolver(
            httpx_client=self.httpx_client,
            base_url=self.a2a_base_url,
        )

        try:
            logger.info(f"Fetching agent card from {self.a2a_base_url}")
            self.agent_card = await resolver.get_agent_card()
            logger.info("Successfully fetched agent card")
            self.a2a_client = A2AClient(
                httpx_client=self.httpx_client, agent_card=self.agent_card
            )
            logger.info("A2A client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize A2A client: {e}")
            raise

    async def call_a2a_agent(
        self, query: str, context_id: str = None, task_id: str = None
    ) -> tuple[str, str, str]:
        """Call the A2A agent and return response, context_id, and task_id.

        Args:
            query: The query to send to the A2A agent
            context_id: Optional context ID for multi-turn conversations
            task_id: Optional task ID (not used, kept for API compatibility)

        Returns:
            Tuple of (response_text, context_id, task_id)
        """
        if not self.a2a_client:
            await self.initialize()

        send_message_payload = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": query}],
                "message_id": uuid4().hex,
            },
        }

        # For follow-up queries, use context_id but NOT task_id
        # The task_id becomes invalid once the task is completed (terminal state)
        # Context_id maintains conversation history across new tasks
        if context_id:
            send_message_payload["message"]["context_id"] = context_id
        # NOTE: We intentionally don't send task_id for follow-up queries
        # Once a task completes, reusing its task_id causes "Task is in terminal state: completed" error
        # Each follow-up query starts a new task but maintains context via context_id

        request = SendMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**send_message_payload)
        )

        response = await self.a2a_client.send_message(request)

        # Extract response text from the A2A response
        # Use the same approach as test_client.py - model_dump to get dict
        response_dict = response.model_dump(mode='json', exclude_none=True)
        
        response_text = ""
        new_context_id = context_id
        new_task_id = None  # task_id not used for follow-ups

        # Check if response is an error response FIRST
        # JSONRPCErrorResponse has 'error' attribute, not 'result'
        # This prevents AttributeError when accessing .result on error responses
        is_error = False
        error_message = ""
        
        # Helper function to extract error message
        def extract_error_message(error_obj):
            """Extract error message from error object."""
            if hasattr(error_obj, 'message'):
                return error_obj.message
            elif isinstance(error_obj, dict):
                return error_obj.get('message', error_obj.get('data', 'Unknown error'))
            else:
                return str(error_obj)
        
        # Check for error in response object
        if hasattr(response, 'root') and response.root:
            root_type_name = type(response.root).__name__
            # Check if it's an error response type or has error attribute
            if hasattr(response.root, 'error') or 'Error' in root_type_name:
                is_error = True
                if hasattr(response.root, 'error'):
                    error_message = extract_error_message(response.root.error)
                else:
                    error_message = f"Error response (type: {root_type_name})"
                logger.warning(f"A2A agent returned an error: {error_message}")
        
        # Fallback: Check in dict structure
        if not is_error and isinstance(response_dict, dict):
            root_data = response_dict.get('root', {})
            if 'error' in root_data:
                is_error = True
                error_message = extract_error_message(root_data['error'])
                logger.warning(f"A2A agent returned an error: {error_message}")

        if is_error:
            # Return error message as response text
            response_text = f"Error from A2A agent: {error_message}"
            return response_text, new_context_id, new_task_id

        # Extract context_id from response (same as test_client.py)
        # Only access .result if it's not an error response and result exists
        if response.root and hasattr(response.root, 'result') and response.root.result:
            result = response.root.result
            if hasattr(result, "context_id") and result.context_id:
                new_context_id = result.context_id
            # NOTE: We don't extract or reuse task_id for follow-up queries
            # Each query creates a new task, but context_id maintains conversation continuity

        # Extract text from artifacts using dict structure (same pattern as test_client)
        def extract_text_from_artifacts(artifacts_data):
            """Extract text from artifacts data structure."""
            text_parts = []
            for artifact in artifacts_data:
                # Get parts from artifact (dict or object)
                if isinstance(artifact, dict):
                    parts = artifact.get('parts', [])
                elif hasattr(artifact, 'parts'):
                    parts = artifact.parts
                else:
                    continue
                
                for part in parts:
                    # Get root from part (dict or object)
                    if isinstance(part, dict):
                        root = part.get('root', {})
                    elif hasattr(part, 'root'):
                        root = part.root
                    else:
                        continue
                    
                    # Extract text from root
                    if isinstance(root, dict) and 'text' in root:
                        text_parts.append(root['text'])
                    elif hasattr(root, 'text') and root.text:
                        text_parts.append(root.text)
            
            return "\n".join(text_parts) if text_parts else ""
        
        # Primary method: Extract from dict structure
        try:
            if isinstance(response_dict, dict):
                result_data = response_dict.get('root', {}).get('result', {})
                artifacts = result_data.get('artifacts', [])
                response_text = extract_text_from_artifacts(artifacts)
        except Exception as e:
            logger.debug(f"Error extracting from dict artifacts: {e}")

        # Fallback: Try direct attribute access
        if not response_text and response.root and hasattr(response.root, 'result') and response.root.result:
            result = response.root.result
            if hasattr(result, "artifacts") and result.artifacts:
                response_text = extract_text_from_artifacts(result.artifacts)

        # Last resort: Use regex to extract from JSON string
        if not response_text:
            try:
                response_str = json.dumps(response_dict)
                text_matches = re.findall(r'"text"\s*:\s*"([^"]+)"', response_str)
                if text_matches:
                    response_text = "\n".join(text_matches)
            except Exception as e:
                logger.debug(f"Error extracting with regex: {e}")

        if not response_text:
            response_text = "Unable to extract response text from A2A agent"

        return response_text.strip(), new_context_id, new_task_id

    def build_persona_graph(self) -> StateGraph:
        """Build the LangGraph for persona interaction with A2A agent."""

        async def prepare_query(state: PersonaState) -> PersonaState:
            """Prepare the query based on persona and conversation history."""
            persona_config = state["persona_config"]
            messages = state["messages"]

            # Build context from conversation
            conversation_context = ""
            if len(messages) > 1:
                conversation_context = "\n\nPrevious conversation:\n"
                for msg in messages[:-1]:
                    if isinstance(msg, HumanMessage):
                        conversation_context += f"Persona: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        conversation_context += f"A2A Agent: {msg.content}\n"

            # Determine what to ask next
            if state["query_count"] == 0:
                # First query - use initial query from persona
                query = persona_config["initial_query"]
            else:
                # Follow-up query - use LLM to generate based on persona and conversation
                prompt = f"""You are {persona_config['name']}.

{persona_config['system_instruction']}

{conversation_context}

Based on the conversation above, what would you like to ask next? 
Generate a follow-up question that aligns with your persona and goals.
Keep it concise and focused on one specific aspect you want to explore further.

Your next question:"""

                llm_response = self.model.invoke([("user", prompt)])
                query = llm_response.content

            return {
                **state,
                "messages": state["messages"] + [HumanMessage(content=query)],
            }

        async def call_a2a_node(state: PersonaState) -> PersonaState:
            """Call the A2A agent with the current query."""
            last_message = state["messages"][-1]
            query = last_message.content

            logger.info(
                f"[{state['persona']}] Querying A2A agent: {query[:100]}..."
            )

            response_text, context_id, task_id = await self.call_a2a_agent(
                query, state.get("context_id"), state.get("task_id")
            )

            logger.info(
                f"[{state['persona']}] Received response: {response_text[:100]}..."
            )

            return {
                **state,
                "a2a_response": response_text,
                "context_id": context_id,
                "task_id": task_id,
                "messages": state["messages"] + [AIMessage(content=response_text)],
                "query_count": state["query_count"] + 1,
            }

        def should_continue(state: PersonaState) -> Literal["continue", "end"]:
            """Decide whether to continue the conversation or end."""
            if state["query_count"] >= state["max_queries"]:
                return "end"

            # Check if response is empty or looks like an error
            last_response = state["a2a_response"]
            if not last_response or len(last_response.strip()) == 0:
                logger.warning("Empty response received, ending conversation")
                return "end"
            
            # If response looks like raw JSON (contains 'jsonrpc' or 'result'), extraction may have failed
            if "jsonrpc" in last_response.lower() or ("'result'" in last_response and len(last_response) > 500):
                logger.warning("Response appears to be raw JSON, extraction may have failed")
                # Still try to continue, but be cautious
                if state["query_count"] >= 2:  # Don't loop too much on bad responses
                    return "end"

            # Use LLM to decide if persona wants to continue
            persona_config = state["persona_config"]

            # Limit response length for prompt (first 1000 chars)
            response_preview = last_response[:1000] if len(last_response) > 1000 else last_response
            
            decision_prompt = f"""You are {persona_config['name']}.

{persona_config['system_instruction']}

You just received this response from an AI agent:
{response_preview}

Based on your persona and goals, do you want to ask a follow-up question?
Respond with 'YES' if you want to ask more, or 'NO' if you're satisfied.

Your decision:"""

            try:
                decision_response = self.model.invoke([("user", decision_prompt)])
                decision = decision_response.content.strip().upper()
                return "continue" if "YES" in decision else "end"
            except Exception as e:
                error_msg = str(e)
                # Check if it's an API key error
                if "401" in error_msg or "api_key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                    logger.error(
                        "API key error detected. Please update OPENAI_API_KEY in your .env file with a valid key."
                    )
                else:
                    logger.warning(f"Error in decision making: {e}, defaulting to end")
                return "end"

        # Build the graph
        graph = StateGraph(PersonaState)

        graph.add_node("prepare_query", prepare_query)
        graph.add_node("call_a2a", call_a2a_node)

        graph.add_edge(START, "prepare_query")
        graph.add_edge("prepare_query", "call_a2a")
        graph.add_conditional_edges(
            "call_a2a",
            should_continue,
            {"continue": "prepare_query", "end": END},
        )

        return graph.compile()

    async def run_persona(self, persona_name: str) -> Dict[str, Any]:
        """Run a specific persona through the conversation.

        Args:
            persona_name: Name of the persona to run ('ml_expert', 'business_analyst', or 'student')

        Returns:
            Dictionary with conversation results
        """
        if persona_name not in PERSONAS:
            raise ValueError(
                f"Unknown persona: {persona_name}. Choose from {list(PERSONAS.keys())}"
            )

        persona_config = PERSONAS[persona_name]

        initial_state: PersonaState = {
            "messages": [],
            "persona": persona_name,
            "persona_config": persona_config,
            "a2a_response": "",
            "context_id": None,
            "task_id": None,
            "query_count": 0,
            "max_queries": self.max_queries_per_persona,
        }

        graph = self.build_persona_graph()

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting conversation with {persona_config['name']}")
        logger.info(f"{'='*60}\n")

        final_state = {"messages": []}
        async for state in graph.astream(initial_state):
            # Collect final state
            if "call_a2a" in state:
                final_state = state["call_a2a"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Conversation with {persona_config['name']} completed")
        logger.info(f"Total queries: {final_state.get('query_count', 0)}")
        logger.info(f"{'='*60}\n")

        return {
            "persona": persona_name,
            "persona_name": persona_config["name"],
            "messages": final_state.get("messages", []),
            "query_count": final_state.get("query_count", 0),
        }

    async def run_all_personas(self) -> List[Dict[str, Any]]:
        """Run all personas and return their conversation results."""
        results = []
        for persona_name in PERSONAS.keys():
            try:
                result = await self.run_persona(persona_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Error running persona {persona_name}: {e}")
                results.append(
                    {
                        "persona": persona_name,
                        "error": str(e),
                    }
                )
        return results

    async def close(self):
        """Close the HTTP client."""
        if self.httpx_client:
            await self.httpx_client.aclose()


async def main():
    """Main function to run the simple agent with all personas."""
    print("🚀 Simple Agent with 3 Personas")
    print("=" * 60)
    print("\nThis agent will demonstrate 3 different personas interacting")
    print("with the A2A agent through LangGraph.\n")

    agent = SimpleAgent(max_queries_per_persona=3)

    try:
        await agent.initialize()

        # Run all personas
        results = await agent.run_all_personas()

        # Print summary with conversation details
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for result in results:
            if "error" in result:
                print(f"❌ {result['persona']}: Error - {result['error']}")
                print()
            else:
                persona_name = result.get('persona_name', result['persona'])
                query_count = result.get('query_count', 0)
                messages = result.get('messages', [])
                
                print(f"✅ {persona_name}")
                print(f"   Queries made: {query_count}")
                
                # Print conversation summary
                if messages:
                    print(f"   Conversation messages: {len(messages)}")
                    print("\n   Conversation Preview:")
                    for i, msg in enumerate(messages, 1):
                        role = "Persona" if msg.__class__.__name__ == "HumanMessage" else "A2A Agent"
                        preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                        print(f"   {i}. [{role}]: {preview}")
                print()

    except Exception as e:
        logger.error(f"Error running simple agent: {e}", exc_info=True)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())

