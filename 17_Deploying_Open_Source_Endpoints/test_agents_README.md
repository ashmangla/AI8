# Test Agents Notebook

This notebook (`test_agents.ipynb`) provides comprehensive testing for three agent implementations powered by **Together AI open-source endpoints**.

## Overview

The notebook tests three different agent architectures:

1. **Simple Agent** - A basic tool-using agent that can call tools and respond to queries
2. **Agent with Helpfulness** - An enhanced agent with a helpfulness evaluation loop that iteratively improves responses
3. **RAG Tool** - A Retrieval-Augmented Generation tool that queries local documents

## Prerequisites

### Required Environment Variables

- `TOGETHER_API_KEY` - Your Together AI API key (required)
- `TOGETHER_MODEL` - The model endpoint to use (defaults to `openai/gpt-oss-20b`)
- `OPENAI_API_KEY` - Required for embeddings in the RAG tool (optional if not using RAG)
- `TAVILY_API_KEY` - Required for Tavily search tool (optional if not using search)

### Dependencies

All dependencies are listed in `pyproject.toml`. Install them with:

```bash
uv sync
```

Or if using pip:

```bash
pip install -r requirements.txt
```

### Data Files

For RAG testing, ensure you have PDF files in the `data/` directory. The notebook will attempt to load and index these documents.

## Using Together AI Endpoints

This notebook uses **Together AI's open-source model endpoints** instead of traditional OpenAI endpoints. The models are configured to use:

- **Model**: `openai/gpt-oss-20b` (default, can be overridden via `TOGETHER_MODEL`)
- **Provider**: Together AI via `langchain-together`
- **Endpoint Type**: Serverless or dedicated endpoints

### Setting Up Your Endpoint

1. Get your Together AI API key from [api.together.ai/settings/api-keys](https://api.together.ai/settings/api-keys)
2. Set the `TOGETHER_API_KEY` environment variable or enter it when prompted
3. Optionally set `TOGETHER_MODEL` to use a custom endpoint (e.g., your dedicated endpoint)

## Notebook Structure

### Cell 1: Setup and Imports
- Loads environment variables
- Prompts for API key if not set
- Imports all necessary modules and graphs

### Test 1: Simple Agent
- **Test 1.1**: Simple query without tool usage
- **Test 1.2**: Query that triggers tool usage (Tavily search, Arxiv)

### Test 2: Agent with Helpfulness
- **Test 2.1**: Query that should produce a helpful response
- **Test 2.2**: Complex query that may require iteration

### Test 3: RAG Tool
- **Test 3.1**: Direct RAG tool invocation
- **Test 3.2**: RAG tool used through the simple agent

### Test 4: Comparison Test
- Runs the same query through all three approaches
- Compares response lengths, message counts, and behavior

## Running the Notebook

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Open** `test_agents.ipynb`

3. **Run cells sequentially**:
   - Start with Cell 1 (Setup) to configure your environment
   - Run each test cell to see the results
   - Review the comparison test to see differences between approaches

## Expected Outputs

### Simple Agent
- Direct responses to queries
- Tool calls when needed (search, arxiv, RAG)
- Final response with tool results integrated

### Agent with Helpfulness
- Initial response from the agent
- Helpfulness evaluation (Y/N)
- Iterative improvements if not helpful
- Final evaluated response

### RAG Tool
- Document retrieval from local PDFs
- Context-aware responses based on retrieved documents
- "I don't know" responses when context is insufficient

## Troubleshooting

### ModuleNotFoundError
If you see import errors, ensure all dependencies are installed:
```bash
uv sync
```

### API Key Errors
- Verify your `TOGETHER_API_KEY` is set correctly
- Check that your Together AI account has sufficient credits
- Ensure the endpoint is active (for dedicated endpoints)

### RAG Tool Errors
- Ensure PDF files exist in the `data/` directory
- Verify `OPENAI_API_KEY` is set for embeddings
- Check that `qdrant-client` is installed

### Arxiv Tool Errors
- Arxiv API has rate limits (429 errors are common)
- The agent will fall back to other tools if Arxiv fails

## Notes

- The notebook uses **Together AI endpoints** for all LLM interactions
- Tool calls (Tavily, Arxiv) may have rate limits
- RAG requires local PDF files and OpenAI embeddings
- Helpfulness agent may iterate multiple times before finding a helpful response
- All agents share the same tool belt (Tavily, Arxiv, RAG)

## Architecture

```
┌─────────────────┐
│  Simple Agent   │ → Uses tools → Returns response
└─────────────────┘

┌─────────────────────────┐
│ Agent w/ Helpfulness    │ → Uses tools → Evaluates → Iterates if needed
└─────────────────────────┘

┌──────────┐
│ RAG Tool │ → Retrieves docs → Generates context-aware response
└──────────┘
```

All agents use the Together AI `openai/gpt-oss-20b` model endpoint by default.

