"""Test script to run just one persona from simple_agent."""
import asyncio
from simple_agent import SimpleAgent

async def main():
    """Test with just one persona."""
    agent = SimpleAgent(max_queries_per_persona=3)
    
    try:
        await agent.initialize()
        # Test just the ML expert persona
        result = await agent.run_persona("ml_expert")
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Persona: {result['persona_name']}")
        print(f"Queries made: {result['query_count']}")
        print(f"Messages: {len(result['messages'])}")
        print("\nConversation:")
        for i, msg in enumerate(result['messages'], 1):
            role = "Persona" if msg.__class__.__name__ == "HumanMessage" else "A2A Agent"
            preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            print(f"{i}. {role}: {preview}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await agent.close()

if __name__ == "__main__":
    asyncio.run(main())

