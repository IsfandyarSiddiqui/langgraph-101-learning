# LangGraph 101

Welcome to LangGraph 101! 

## Introduction
This repository contains hands-on tutorials for learning LangChain, LangGraph, and Deep Agents, organized into two learning tracks:

- **101**: Fundamentals of building agents with LangChain and LangGraph
- **201**: Advanced patterns including multi-agent systems, deep agents, and production workflows

To understand how these frameworks relate to each other, see [LangChain vs LangGraph vs Deep Agents](https://docs.langchain.com/oss/python/concepts/products).

This is a condensed version of LangChain Academy, intended to be run in a session with a LangChain engineer. If you're interested in going into more depth, or working through tutorials on your own, check out [LangChain Academy](https://academy.langchain.com/courses/intro-to-langgraph)! LangChain Academy has helpful pre-recorded videos from our LangChain engineers.

## What's Inside

### 101 - Fundamentals (`notebooks/101/`)
- **101_langchain_langgraph.ipynb**: Build your first agent with models, tools, memory, and streaming
- **102_middleware.ipynb**: Middleware, human-in-the-loop patterns, and guardrails

### 201 - Production Patterns (`notebooks/201/`)
- **email_agent.ipynb**: Build a stateful email triage and response agent
- **multi_agent.ipynb**: Multi-agent systems with supervisors and specialized sub-agents
- **research_agent.ipynb**: Deep research agent with parallel sub-researchers
- **deepagents.ipynb**: Build a research agent from scratch with DeepAgents -- covers AGENTS.md, skills, backends, long-term memory, HITL, and more

### Agents (`agents/`)
Standalone agent implementations that run in LangGraph Studio via `langgraph dev`:
- **`agents/101/`** - Simple weather agent from the 101 notebook
- **`agents/email_agent/`** - Email triage agent
- **`agents/music_store/`** - Multi-agent music store (supervisor + subagents)
- **`agents/researcher/`** - Deep research agent with parallel sub-researchers
- **`agents/deep_agent/`** - DeepAgents research agent with AGENTS.md, skills (LinkedIn post, Twitter/X post), long-term memory, and HITL

All notebooks use the latest **LangChain**, **LangGraph**, and **DeepAgents** primitives, including `create_agent()`, `create_deep_agent()`, middleware, and interrupt patterns.

## Project Structure

```
langgraph-101/
├── notebooks/
│   ├── 101/                          # Fundamentals
│   │   ├── 101_langchain_langgraph.ipynb
│   │   └── 102_middleware.ipynb
│   └── 201/                          # Production Patterns
│       └── deep_agents.ipynb
│       ├── email_agent.ipynb
│       ├── multi_agent.ipynb
│       ├── research_agent.ipynb

├── agents/                           # Standalone agents for LangGraph Studio
│   ├── 101/agent.py
│   ├── email_agent/graph.py
│   ├── music_store/                  # Multi-agent supervisor + subagents
│   ├── researcher/                   # Deep research agent
│   └── deep_agent/                   # DeepAgents research agent
│       ├── agent.py                  # Agent definition
│       ├── AGENTS.md                 # Agent identity & instructions
│       └── skills/                   # On-demand capabilities
│           ├── linkedin-post/SKILL.md
│           └── twitter-post/SKILL.md
├── utils/
│   ├── models.py                     # Centralized model configuration
│   └── utils.py                      # Shared utilities
├── langgraph.json                    # Agent registry for langgraph dev
└── .env                              # API keys (not committed)
```

## Context

At LangChain, we aim to make it easy to build LLM applications. One type of LLM application you can build is an agent. There's a lot of excitement around building agents because they can automate a wide range of tasks that were previously impossible. 

In practice though, it is incredibly difficult to build systems that reliably execute on these tasks. As we've worked with our users to put agents into production, we've learned that more control is often necessary. You might need an agent to always call a specific tool first or use different prompts based on its state.

To tackle this problem, we've built [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — a framework for building agent and multi-agent applications. Separate from the LangChain package, LangGraph's core design philosophy is to help developers add better precision and control into agent workflows, suitable for the complexity of real-world systems.

For complex, multi-step tasks that require planning, filesystem access, and delegation, we've built [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) — an agent harness on top of LangGraph that provides built-in tools, context management, and skills out of the box.

## Pre-work

### Clone the LangGraph 101 repo
```
git clone https://github.com/langchain-ai/langgraph-101.git
```


### Create an environment 
Ensure you have a recent version of pip and python installed
```
$ cd langgraph-101
# Copy the .env.example file to .env
cp .env.example .env
```
If you run into issues with setting up the python environment or acquiring the necessary API keys due to any restrictions (ex. corporate policy), contact your LangChain representative and we'll find a work-around!

### Package Installation
Ensure you have a recent version of pip and python installed
```
# Install uv if you haven't already
pip install uv

# Install the package, allowing for pre-release 
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### Running Agents Locally

You can run the agents in this repository locally using `langgraph dev`. This gives you:
- A local API server for your agents
- LangGraph Studio UI for testing and debugging
- Hot-reloading during development

```bash
# From the root directory, start the LangGraph development server
langgraph dev

# This will start a local server and provide:
# - API endpoint for your agents (typically http://localhost:2024)
# - LangGraph Studio UI link
```

The `langgraph.json` configuration file defines which agents are available. You can interact with agents via the API or through LangGraph Studio's visual interface.

For more details, see the [LangGraph CLI documentation](https://docs.langchain.com/langsmith/cli#langgraph-cli).

### Model Configuration

This repository uses a **centralized utils module** (`utils/`) to avoid code duplication. All model configurations and shared utilities are defined here:
- **`utils/models.py`** - LLM model initialization (OpenAI, Anthropic, Groq, Google, Vertex AI, NVIDIA)
- **`utils/utils.py`** - Shared utility functions (`show_graph`, `get_engine_for_chinook_db`)

**Default**: OpenAI with `gpt-4.1-mini`.

**Note**: Notebooks automatically add the project root to Python's path, so they can import from `utils` regardless of which subdirectory they're in.

To switch models, set `MODEL_PROVIDER` to a key in the `PROVIDERS` dictionary.
Set `MODEL_NAME` to one of that provider class's models, or omit it to use the
last (most capable) model:

```bash
MODEL_PROVIDER="grok"
MODEL_NAME="grok-4.5"
GROK_API_KEY="your-api-key"
```

For local development, put these values in `.env`. For GitHub Actions or
Codespaces, configure them as secrets/environment variables. GitHub exposes
secrets as environment variables, so application code uses the same interface
in both environments; `IS_GITHUB_ENV` prevents a local `.env` from replacing
GitHub-provided values.

Each provider class owns its LangChain class, API-key variable, models in
ascending capability order, and provider-specific constructor options.

## Getting Started

### Recommended Learning Path

1. **Start with 101** - `notebooks/101/`
   - Begin with `101_langchain_langgraph.ipynb` to learn LangChain + LangGraph fundamentals
   - Continue with `102_middleware.ipynb` for middleware and human-in-the-loop patterns

2. **Progress to 201** - `notebooks/201/`
   - Explore `email_agent.ipynb` for a complete stateful agent example
   - Build multi-agent systems with `multi_agent.ipynb`
   - Learn Deep Agents with `deepagents.ipynb` -- progressively build a research agent with AGENTS.md, skills, backends, memory, and HITL

3. **Run Agents in Studio**
   - Use `langgraph dev` to launch all agents in LangGraph Studio
   - Try the Deep Agent (`agents/deep_agent/`) -- ask it to research a topic and write a LinkedIn post

### Resources

- **[LangChain Documentation](https://docs.langchain.com/oss/python/langchain/overview)** - Complete LangChain reference
- **[LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview)** - LangGraph guides and API reference
- **[Deep Agents Documentation](https://docs.langchain.com/oss/python/deepagents/)** - Deep Agents harness reference
- **[LangChain vs LangGraph vs Deep Agents](https://docs.langchain.com/oss/python/concepts/products)** - How the frameworks relate
- **[LangChain Academy](https://academy.langchain.com/)** - Free courses with video tutorials
- **[LangSmith](https://smith.langchain.com)** - Debugging and monitoring for LLM applications
