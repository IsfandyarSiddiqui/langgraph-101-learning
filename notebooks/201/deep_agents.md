# Deep Agents: Building a Research Agent from Scratch

Welcome to Deep Agents! This notebook will walk you through the core concepts of the Deep Agents framework by **progressively building a research agent** from scratch.

**What you'll learn:**
- What Deep Agents is and what it provides out of the box
- How to add custom tools to extend agent capabilities
- Understanding backends and storage abstraction
- Task delegation with subagents and context isolation
- Human-in-the-loop patterns for safety
- Long-term memory for persistent storage
- Middleware architecture and extensibility
- Skills for reusable agent capabilities

<br>
<br>
---
<br>

> **Note:** This workshop requires a [Tavily API key](https://tavily.com) for web search functionality. Deep Agents is built on top of LangGraph, providing a powerful harness for building autonomous agents with filesystem access, planning, and delegation capabilities.

## Part 0: Setup & Installation

First, let's install the necessary packages and set up our environment.

```python
# Install required packages 
# Run uv sync to install the packages or run:
# !pip install deepagents tavily-python python-dotenv
```

### Initialize your LLM

```python
# Add project root to Python path so we can import from utils module
import sys
from pathlib import Path
project_root = Path().resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import model from centralized utils module
from utils.models import model

# Load environment variables for Tavily
from dotenv import load_dotenv
load_dotenv(dotenv_path="../../.env", override=True)

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore', message='LangSmith now uses UUID v7')
```

## Part 1: Your First Deep Agent (The Harness)

<img src="../../images/deepAgentsDiag.png" style="width: auto; height: 500px; border-radius: 8px;" alt="Deep Agents Architecture">

Deep Agents functions as an **"agent harness"**—a framework built on a core tool-calling loop, but with pre-built tools and integrated capabilities for autonomous task execution.

### What you get out of the box:

- **Filesystem Tools** — `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`
- **Planning Tool** — `write_todos` for task tracking
- **Subagent Delegation** — `task()` tool for isolated work
- **Large Tool Result Eviction** — Automatically offloads tool results >20k tokens to the filesystem
- **Conversation Summarization** — Compresses history when approaching ~85% context capacity
- **Dangling Tool Call Patching** — Fixes message history consistency automatically

Let's create the most basic deep agent:

```python
from deepagents import create_deep_agent

# Create the most basic deep agent - just a model!
agent = create_deep_agent(
    model=model,
    system_prompt="You are a helpful research assistant. When referencing file paths, use backtick formatting like `path/file.md` instead of markdown links."
)

print("Deep agent created successfully!")
```

### Test the built-in filesystem tools

Even without adding any custom tools, our agent already has filesystem capabilities:

```python
# Ask the agent to write and read a file
result = agent.invoke({
    "messages": [{"role": "user", "content": "Write a file called notes.md with the text 'Hello from Deep Agents!' then read it back to confirm."}]
})

# Print the final response
print("Agent response:", result["messages"][-1].content)
```

```python
# Show the virtual filesystem contents (stored in result["files"])
print("\n" + "=" * 50)
print("📁 VIRTUAL FILESYSTEM (in-memory, not on disk!)")
print("=" * 50)
for path, file_data in result.get("files", {}).items():
    print(f"\n  Path: '{path}'")
    print("  " + "-" * 38)
    # file_data contains 'content' (list of lines), 'created_at', 'modified_at'
    if isinstance(file_data, dict) and "content" in file_data:
        content = "\n".join(file_data["content"])
        for line in content.split("\n"):
            print(f"  | {line}")
    else:
        print(f"  | {file_data}")
```

### Key Takeaway:
- `create_deep_agent()` gives you filesystem + planning capabilities for free
- No need to define basic file operations - they're built-in
- Files are stored in agent state (we'll learn more about this in Part 3)

## Part 2: Adding Custom Tools

While the built-in tools are powerful, we need **custom tools** to build a research agent. Let's add web search and strategic thinking capabilities.

### Creating a Web Search Tool with Tavily

```python
from langchain_core.tools import tool
from tavily import TavilyClient

# Initialize Tavily client
tavily_client = TavilyClient()


@tool(parse_docstring=True)
def tavily_search(query: str) -> str:
    """Search the web for information on a given query.
    
    Args:
        query: Search query to execute
    """
    search_results = tavily_client.search(query, max_results=3, topic="general")
    
    result_texts = []
    for result in search_results.get("results", []):
        url = result["url"]
        title = result["title"]
        content = result.get("content", "No content available")
        result_text = f"## {title}\n**URL:** {url}\n\n{content}\n\n---\n"
        result_texts.append(result_text)
    
    return f"Found {len(result_texts)} result(s) for '{query}':\n\n{''.join(result_texts)}"


print("tavily_search tool created!")
```

### Add custom tools to our agent

```python
# Recreate the agent with our custom tools
agent = create_deep_agent(
    model=model,
    tools=[tavily_search],  # Add our custom tools
    system_prompt="""You are a helpful research assistant.
    
Use tavily_search to find information on the web.
When referencing file paths, use backtick formatting like `path/file.md` instead of markdown links.
"""
)

print("Agent updated with custom tools!")
```

```python
# Test the search capability
result = agent.invoke({
    "messages": [{"role": "user", "content": "Search for information about LangGraph and summarize what you find."}]
})

print(result["messages"][-1].content)
```

### Key Takeaway:
- Custom tools extend what your agent can do
- The `@tool` decorator converts a function into a LangChain tool

## Part 3: Understanding Backends

<img src="../../images/deepAgentBackends.png" style="width: auto; height: 500px; border-radius: 8px;" alt="Backend Architecture">

Where do the agent's files actually go? **Backends** are pluggable storage systems that expose a filesystem surface to agents.

### Four Backend Types:

| Backend | Storage | Persistence | Use Case |
|---------|---------|-------------|----------|
| **StateBackend** | In-memory (agent state) | Single thread | Scratch pads, intermediate results |
| **FilesystemBackend** | Local disk | Permanent | Direct file access (use with caution!) |
| **StoreBackend** | LangGraph Store | Cross-thread | Long-term memories |
| **CompositeBackend** | Routes to others | Mixed | Selective persistence |

By default, `create_deep_agent()` uses **StateBackend** — files are stored in agent state and disappear when the thread ends.

```python
from langgraph.checkpoint.memory import MemorySaver
from langsmith import uuid7

# Add a checkpointer so we can demonstrate persistence across turns
checkpointer = MemorySaver()

agent = create_deep_agent(
    model=model,
    tools=[tavily_search],
    system_prompt="You are a helpful research assistant. When referencing file paths, use backtick formatting like `path/file.md` instead of markdown links.",
    checkpointer=checkpointer
)

# Create a thread
thread_id = str(uuid7())
config = {"configurable": {"thread_id": thread_id}}

print(f"Thread ID: {thread_id}")
```

```python
# Write a file in this thread
result = agent.invoke({
    "messages": [{"role": "user", "content": "Write a file called /research_notes.md with 'My research findings go here'"}]
}, config=config)

print("Files in state:", list(result.get("files", {}).keys()))
```

```python
# In the same thread, the file persists
result = agent.invoke({
    "messages": [{"role": "user", "content": "Read the file `/research_notes.md`"}]
}, config=config)

print(result["messages"][-1].content)
```

```python
# In a NEW thread, the file is gone (StateBackend is ephemeral)
new_config = {"configurable": {"thread_id": str(uuid7())}}

result = agent.invoke({
    "messages": [{"role": "user", "content": "List all files with ls /"}]
}, config=new_config)

print(result["messages"][-1].content)
```

### FilesystemBackend - Writing to Real Disk

When you need agents to work with **actual files on disk**, use `FilesystemBackend`. With `virtual_mode=True`, paths are sandboxed under `root_dir` for security.

> ⚠️ **Caution**: FilesystemBackend writes real files! Use `virtual_mode=True` to prevent path traversal attacks.

```python
from deepagents.backends import FilesystemBackend
import tempfile
import os

# Create a temporary directory for our sandbox
sandbox_dir = tempfile.mkdtemp(prefix="deepagents_sandbox_")
print(f"Sandbox directory: `{sandbox_dir}`")

# Create a FilesystemBackend with virtual_mode=True (sandboxed)
fs_backend = FilesystemBackend(root_dir=sandbox_dir, virtual_mode=True)

# Create an agent that writes to real disk (sandboxed)
agent_with_fs = create_deep_agent(
    model=model,
    system_prompt="You are a helpful assistant. Files you write go to the local filesystem. When referencing file paths, use backtick formatting like `path/file.md` instead of markdown links.",
    backend=fs_backend,
    checkpointer=checkpointer
)

print("Agent with FilesystemBackend created!")
```

```python
# Test: Write a file through the agent
config = {"configurable": {"thread_id": str(uuid7())}}

result = agent_with_fs.invoke({
    "messages": [{"role": "user", "content": "Write a file called notes.txt with 'Hello from FilesystemBackend!'"}]
}, config=config)

print("Agent response:", result["messages"][-1].content)

# Verify the file was actually written to disk!
actual_path = os.path.join(sandbox_dir, "notes.txt")
if os.path.exists(actual_path):
    with open(actual_path, "r") as f:
        print(f"\n✅ File exists on disk at: `{actual_path}`")
        print(f"   Content: {f.read()}")
else:
    print(f"\n❌ File not found at: {actual_path}")

# List all files in the sandbox
print(f"\n📁 Files in sandbox ({sandbox_dir}):")
for f in os.listdir(sandbox_dir):
    print(f"   - {f}")
```

### CompositeBackend - Routing Paths to Different Backends

`CompositeBackend` lets you route different paths to different backends. This is how you implement **hybrid storage** - some paths ephemeral, others persistent, others on disk.

```
/                 → StateBackend (ephemeral scratch space)
/memories/*       → StoreBackend (persistent across threads)
/workspace/*      → FilesystemBackend (real disk)
```

```python
from deepagents.backends import StateBackend, CompositeBackend

# Create another sandbox for workspace files
workspace_dir = tempfile.mkdtemp(prefix="deepagents_workspace_")
print(f"Workspace directory: `{workspace_dir}`")

composite_backend = CompositeBackend(
    default=StateBackend(),
    routes={
        # /workspace/* → FilesystemBackend (real disk, sandboxed)
        "/workspace/": FilesystemBackend(root_dir=workspace_dir, virtual_mode=True),
    }
)

agent_composite = create_deep_agent(
    model=model,
    system_prompt="""You are a helpful assistant.

STORAGE RULES:
- Files in /workspace/* are saved to real disk (persistent)
- All other files are ephemeral (disappear when thread ends)

When referencing file paths, use backtick formatting like `path/file.md` instead of markdown links.
""",
    backend=composite_backend,
    checkpointer=checkpointer
)

print("Agent with CompositeBackend created!")
```

```python
# Test: Write to both locations and see the difference
config = {"configurable": {"thread_id": str(uuid7())}}

result = agent_composite.invoke({
    "messages": [{"role": "user", "content": """Write two files:
1. `/workspace/persistent.txt` with 'I will survive!'
2. `/scratch.txt` with 'I am ephemeral'

Then list both locations."""}]
}, config=config)

print("Agent response:", result["messages"][-1].content)

# Check what's on disk vs in state
print("\n" + "=" * 50)
print("📁 ON DISK (workspace_dir):")
for f in os.listdir(workspace_dir):
    filepath = os.path.join(workspace_dir, f)
    with open(filepath, "r") as file:
        print(f"   {f}: {file.read()}")

print("\n📦 IN VIRTUAL STATE:")
for path, data in result.get("files", {}).items():
    if isinstance(data, dict) and "content" in data:
        content = "\n".join(data["content"])
    else:
        content = str(data)
    print(f"   `{path}`: {content}")
```

### StoreBackend - For LangSmith Deployment

`StoreBackend` uses LangGraph's `BaseStore` for persistent, cross-thread storage. This is the backend you'll use when:

- Running `langgraph dev` locally (store is provided automatically)
- Deploying to **LangSmith Deployments** (store is managed by the platform)

The platform provides a persistent store, so your agent's `/memories/*` files survive across conversations and server restarts.

> 💡 **Note**: We'll demonstrate `StoreBackend` in Part 6 (Long-Term Memory) where we combine it with `CompositeBackend` for selective persistence.

```python
# When deployed to LangSmith Deployment, the store is injected automatically
# This is the pattern you'll use in production:
from deepagents.backends import StateBackend, StoreBackend, CompositeBackend
from langgraph.store.memory import InMemoryStore

agent = create_deep_agent(
    backend=CompositeBackend(
        default=StateBackend(),  # Everything else is ephemeral
        routes={
            # /memories/* persists across threads via the platform's store
            "/memories/": StoreBackend(),
        }
    ),
    store=InMemoryStore()  # Good for local dev; omit for LangSmith Deployment
)
```

```python
# Cleanup: Remove temporary directories
import shutil
shutil.rmtree(sandbox_dir, ignore_errors=True)
shutil.rmtree(workspace_dir, ignore_errors=True)
print("✅ Temporary directories cleaned up")
```

### Key Takeaway:
- **Backends** control where/how agent files are stored
- **StateBackend** (default) is ephemeral - files disappear when thread ends
- **FilesystemBackend** writes to real disk (use `virtual_mode=True` for sandboxing)
- **CompositeBackend** routes different paths to different backends (hybrid storage)
- **StoreBackend** is used for LangSmith Deployment (we'll use it in Part 6)



## Part 4: Adding a Research Subagent

<img src="../../images/deepAgentSubagents.png" style="width: auto; height: 500px; border-radius: 8px;" alt="Subagent Architecture">

As agents do more work, their context fills up with intermediate tool calls. **Subagents** solve this by isolating work in a separate context.

### The Context Bloat Problem

Without subagents:
```
User: Research AI agents
→ search("AI agents overview") → 5000 tokens of results
→ think("Found overview...") → 200 tokens
→ search("AI agent frameworks") → 5000 tokens of results  
→ think("Found frameworks...") → 200 tokens
→ ... context bloats with every search
```

With subagents:
```
User: Research AI agents
→ task("research-agent", "Research AI agents")
→ [Subagent does all searches in isolated context]
→ Returns: "Summary: AI agents are..." (clean, compressed result)
```

```python
from datetime import datetime

# Get current date for the researcher
current_date = datetime.now().strftime("%Y-%m-%d")

# Define the researcher subagent
RESEARCHER_INSTRUCTIONS = f"""You are a research assistant conducting research. Today's date is {current_date}.

<Task>
Use tools to gather information about the research topic.
</Task>

<Hard Limits>
- Simple queries: Use 2-3 search tool calls maximum
- Complex queries: Use up to 5 search tool calls maximum
</Hard Limits>

<Output Format>
Structure your findings with:
- Clear headings
- Inline citations [1], [2], [3]
- Sources section at the end
</Output Format>

When referencing file paths, use backtick formatting like `path/file.md` instead of markdown links.
"""

research_subagent = {
    "name": "research-agent",
    "description": "Delegate research tasks. Give one topic at a time.",
    "system_prompt": RESEARCHER_INSTRUCTIONS,
    "tools": [tavily_search],
}

print("Research subagent defined!")
print(f"  Name: {research_subagent['name']}")
print(f"  Tools: {[t.name for t in research_subagent['tools']]}")
```

```python
# Define orchestrator instructions
ORCHESTRATOR_INSTRUCTIONS = """You are a research coordinator.

When asked to research a topic:
1. Use write_todos to plan your research tasks
2. Delegate research to the research-agent subagent using the task() tool
3. NEVER search directly - always delegate to the research-agent
4. Synthesize findings and write a report to /final_report.md

The research-agent will handle all web searches and return summarized findings.

When referencing file paths, use backtick formatting like `path/file.md` instead of markdown links.
"""

# Create agent with subagent
agent = create_deep_agent(
    model=model,
    tools=[tavily_search],
    system_prompt=ORCHESTRATOR_INSTRUCTIONS,
    subagents=[research_subagent],  # Add our research subagent
    checkpointer=checkpointer
)

print("Agent with subagent created!")
```

```python
# Test delegation
config = {"configurable": {"thread_id": str(uuid7())}}

result = agent.invoke({
    "messages": [{"role": "user", "content": "Lightly research this week's intersesting news on AI agents"}]
}, config=config)

print(result["messages"][-1].content[:2000] + "...")
```

### Key Takeaway:
- Subagents isolate work in a separate context
- Main agent only sees the final result, not intermediate searches
- This keeps the main agent's context clean and focused

## Part 5: Middleware Deep Dive

Deep Agents uses a **modular middleware architecture**. When you call `create_deep_agent()`, several middleware components are automatically attached.
<img src="../../images/deepAgentMiddleware.png" style="width: auto; height: 500px; border-radius: 8px;" alt="Middleware">

### Always-On Middleware:

| Middleware | Tools Provided | Purpose |
|------------|---------------|----------|
| **TodoListMiddleware** | `write_todos` | Task planning and tracking |
| **FilesystemMiddleware** | `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` | File operations + **large tool result eviction** |
| **SubAgentMiddleware** | `task` | Delegate work to subagents |
| **SummarizationMiddleware** | *(none)* | Compresses conversation history at ~85% context capacity |
| **PatchToolCallsMiddleware** | *(none)* | Fixes dangling tool calls in message history |

### Conditional Middleware (added when configured):

| Middleware | Trigger | Purpose |
|------------|---------|----------|
| **MemoryMiddleware** | `memory=["/AGENTS.md"]` | Loads persistent context from AGENTS.md files |
| **SkillsMiddleware** | `skills=["/skills/"]` | Progressive disclosure of bundled capabilities |
| **HumanInTheLoopMiddleware** | `interrupt_on={...}` | Human approval for sensitive operations |

### Built-in Context Management Strategies

Deep agents automatically manage context to stay within the model's token limits. There are three strategies, all configurable via `tool_token_limit_before_evict` and the model's `max_input_tokens`:

<style>
.ctx-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 10px; margin: 10px 0 10px 0; }
.ctx-card img { width: 1000px; border-radius: 8px; margin-bottom: 8px; }
.ctx-card p { font-size: 0.92em; line-height: 1.5; margin: 0; }
.ctx-card strong { display: block; margin-bottom: 4px; }
</style>
<div class="ctx-grid">
<div class="ctx-card">
<img src="../../images/Offloading%20Inputs%20LangChain.png" alt="Offloading tool inputs">
<p><strong>Offload Large Inputs</strong>File write/edit tool calls leave the full content in conversation history. Since it's already on disk, it's redundant. At ~85% context capacity, deep agents truncate these older tool calls and replace them with a pointer to the file.</p>
</div>
<div class="ctx-card">
<img src="../../images/Offloading%20Results%20LangChain.png" alt="Offloading tool results">
<p><strong>Offload Large Results</strong>When a tool result exceeds ~20k tokens, the deep agent saves it to the backend and swaps it with a file path reference + a 10-line preview. The agent can re-read or search the full content as needed.</p>
</div>
<div class="ctx-card">
<img src="../../images/LangChain%20Summarization.png" alt="Conversation summarization">
<p><strong>Conversation Summarization</strong>When context hits ~85% of <code>max_input_tokens</code> and nothing is left to offload, the agent summarizes the history. The full messages are preserved to <code>/conversation_history/</code> on disk; a structured summary replaces them in working memory.</p>
</div>
</div>

```python
# Our agent already has all three middleware!
# Let's see the planning capability in action

config = {"configurable": {"thread_id": str(uuid7())}}

result = agent.invoke({
    "messages": [{"role": "user", "content": "Create a plan for researching machine learning frameworks. Use write_todos."}]
}, config=config)

print(result["messages"][-1].content)
```

```python
# Check the todos in state
if "todos" in result:
    print("📋 Agent's Todo List:\n")
    for todo in result["todos"]:
        status_map = {"completed": "✅", "in_progress": "🔄", "pending": "⬚"}
        status = todo.get("status", "pending")
        icon = status_map.get(status, "⬚")
        content = todo.get("content", str(todo))
        print(f"  {icon} {content}")
else:
    print("No todos in state (agent may have used a different approach)")
```

### Custom Middleware: Logging Tool Calls

Beyond the built-in middleware, you can write your own. The simplest way is with **decorator-based middleware** using hooks like `@wrap_tool_call`, which wraps around every tool invocation.

This is useful for observability — seeing exactly what the agent is doing in real time.

Available hooks:

| Hook | When it runs | Style |
|------|-------------|-------|
| `@before_agent` | Before agent starts (once) | Node |
| `@before_model` | Before each LLM call | Node |
| `@after_model` | After each LLM response | Node |
| `@after_agent` | After agent completes (once) | Node |
| `@wrap_model_call` | Around each LLM call | Wrap |
| `@wrap_tool_call` | Around each tool call | Wrap |

Let's create a simple `@wrap_tool_call` middleware that logs every tool the agent uses:

```python
from langchain.agents.middleware import wrap_tool_call

@wrap_tool_call
def log_tool_calls(request, handler):
    """Log every tool call the agent makes."""
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]
    print(f"🔧 [Tool Call] {tool_name}")
    print(f"   Args: {tool_args}")

    result = handler(request)

    print(f"✅ [Tool Done] {tool_name}\n")
    return result


# Create an agent with our custom middleware
agent_with_logging = create_deep_agent(
    model=model,
    tools=[tavily_search],
    system_prompt="You are a helpful research assistant. When referencing file paths, use backtick formatting like `path/file.md` instead of markdown links.",
    middleware=[log_tool_calls],
    checkpointer=MemorySaver(),
)
```

```python
# Run the agent -- watch the tool call logs in the output!
config = {"configurable": {"thread_id": str(uuid7())}}

result = agent_with_logging.invoke({
    "messages": [{"role": "user", "content": "What is LangGraph? create a short summary in your filesystem."}]
}, config=config)

print("\n--- Agent Response ---")
print(result["messages"][-1].content)
```

### Key Takeaway:
- Middleware = pluggable capability modules that wrap the agent's model calls and tool calls
- `create_deep_agent()` automatically adds TodoList, Filesystem, SubAgent, Summarization, and PatchToolCalls middleware
- **Context management is built-in**: large tool results are evicted, conversation history is summarized
- Writing custom middleware is straightforward — use decorator hooks like `@wrap_tool_call` for quick single-purpose middleware, or class-based `AgentMiddleware` for more complex cases
- Pass custom middleware via `middleware=[...]` and it composes with all the built-in middleware

## Part 6: Human-in-the-Loop

<img src="../../images/deepAgentHITL.png" style="width: auto; height: 500px; border-radius: 8px;" alt="Human-in-the-Loop Flow">

For sensitive operations, you may want a human to approve actions before they execute. Deep Agents has support for **interrupts** built-in via `HumanInTheLoopMiddleware`.

### Built-in Decision Types:
- **Approve** — Execute with proposed arguments
- **Edit** — Modify arguments before execution
- **Reject** — Skip the tool call entirely

Additionally you can also customize which decisions are available for each tool, for example:
```python
interrupt_on = {
    # Sensitive operations: allow all options
    "delete_file": {"allowed_decisions": ["approve", "edit", "reject"]},

    # Moderate risk: approval or rejection only
    "write_file": {"allowed_decisions": ["approve", "reject"]},

    # Must approve (no rejection allowed)
    "critical_operation": {"allowed_decisions": ["approve"]},
}
```

#### For this demo, let's use the default interrupt decision types:

```python
# Create agent with interrupts on file writes
agent_with_hitl = create_deep_agent(
    model=model,
    tools=[tavily_search],
    system_prompt="You are a helpful research assistant. When referencing file paths, use backtick formatting like `path/file.md` instead of markdown links.",
    subagents=[research_subagent],
    checkpointer=checkpointer,
    interrupt_on={
        "write_file": True,  # Interrupt before writing files
        "edit_file": True,   # Interrupt before editing files
    }
)

print("Agent with HITL created!")
```

#### Let's give it a try!

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": str(uuid7())}}

# This will trigger an interrupt when the agent tries to write a file
result = agent_with_hitl.invoke({
    "messages": [{"role": "user", "content": "Write a file called /test.md with 'Hello World'"}]
}, config=config)

# Check if we hit an interrupt
if result.get("__interrupt__"):
    print("🛑 Interrupt triggered!\n")
    interrupt_value = result["__interrupt__"][0].value
    action_requests = interrupt_value["action_requests"]
    review_configs = interrupt_value["review_configs"]

    for action, review in zip(action_requests, review_configs):
        print(f"  Tool: {action['name']}")
        print(f"  Args: {action['args']}")
        print(f"  Allowed decisions: {review['allowed_decisions']}")
else:
    print("No interrupt (file was written)")
    print(result["messages"][-1].content)
```

#### Resume with approval

```python
if result.get("__interrupt__"):
    # Approve the write operation
    result = agent_with_hitl.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config
    )
    print("✅ Resumed with approval!")
    print(result["messages"][-1].content)
```

### Key Takeaway:
- HITL adds human oversight for risky operations
- Configure which tools require approval with `interrupt_on`
- A checkpointer is required for HITL to work

## Part 7: Long-Term Memory

<img src="../../images/deepAgentMemories.png" style="width: auto; height:500px; border-radius: 8px;" alt="Long-Term Memory">

So far, files disappear when threads end. **Long-term memory** uses a `CompositeBackend` to route certain paths to persistent storage.

### Path Routing:
- `/memories/*` → **StoreBackend** (persistent across threads)
- Everything else → **StateBackend** (ephemeral)

```python
from langgraph.store.memory import InMemoryStore
from deepagents.backends import StateBackend, StoreBackend, CompositeBackend

store = InMemoryStore()

composite_backend = CompositeBackend(
    default=StateBackend(),
    routes={
        # Files under /memories/ go to persistent store
        "/memories/": StoreBackend(),
    },
)

print("Composite backend created!")
```

```python
# Create agent with long-term memory
agent_with_memory = create_deep_agent(
    model=model,
    tools=[tavily_search],
    system_prompt="""You are a helpful research assistant with long-term memory.
    
IMPORTANT: Save important notes to /memories/ so they persist across conversations.
For example: /memories/research_notes.md

Regular files (not in /memories/) will disappear when the conversation ends.

When asked what you remember or know, ALWAYS use ls and read_file to check /memories/ first.
Do not rely on conversation context alone -- memories persist across threads but conversation does not.

When referencing file paths, use backtick formatting like `path/file.md` instead of markdown links.
""",
    subagents=[research_subagent],
    checkpointer=checkpointer,
    backend=composite_backend,
    store=store,  # Store is passed here, not to the backend directly
)

print("Agent with long-term memory created!")
```

```python
# Thread 1: Save something to long-term memory
thread1_config = {"configurable": {"thread_id": str(uuid7())}}

result = agent_with_memory.invoke({
    "messages": [{"role": "user", "content": "Save 'Important research finding: AI agents are evolving rapidly' to `/memories/findings.md`"}]
}, config=thread1_config)

print("Thread 1:", result["messages"][-1].content)
```

```python
# Thread 2: Access the memory from a different thread
thread2_config = {"configurable": {"thread_id": str(uuid7())}}

result = agent_with_memory.invoke({
    "messages": [{"role": "user", "content": "Read me what's in the file `/memories/findings.md`"}]
}, config=thread2_config)

print("Thread 2:", result["messages"][-1].content)
```

### Key Takeaway:
- `CompositeBackend` routes different paths to different storage backends
- `/memories/*` persists across threads
- Other files remain ephemeral (single thread)

### Advanced Memory Patterns: Semantic, Episodic & Procedural

Research in cognitive science (and the [CoALA paper](https://arxiv.org/abs/2309.02427)) identifies three types of long-term memory that map naturally to how agents store information:

| Memory Type | What It Stores | Human Example | Agent Example |
|-------------|---------------|---------------|---------------|
| **Semantic** | Facts & knowledge | "Paris is the capital of France" | User preferences, project context |
| **Episodic** | Past experiences | "Last Tuesday I went hiking" | Past research sessions, interaction logs |
| **Procedural** | Instructions & rules | How to ride a bike | Coding standards, report formatting rules |

We can map these to **filesystem paths** using `CompositeBackend` with multiple routes:

```
/memories/semantic/       -> Facts: user_preferences.md, project_context.md
/memories/episodic/       -> Experiences: 2025-02-17_research.md
/memories/procedural/     -> Rules: coding_standards.md, report_format.md
/                         -> Ephemeral scratch space (StateBackend)
```

Since CompositeBackend uses **longest-prefix matching**, `/memories/semantic/` takes priority over a broader `/memories/` route.

```python
# Create a CompositeBackend with routes for each memory type

memory_store = InMemoryStore()

advanced_composite_backend = CompositeBackend(
    default=StateBackend(),
    routes={
        "/memories/semantic/":   StoreBackend(namespace=lambda rt: ("memories", "semantic")),
        "/memories/episodic/":   StoreBackend(namespace=lambda rt: ("memories", "episodic")),
        "/memories/procedural/": StoreBackend(namespace=lambda rt: ("memories", "procedural")),
    },
)

print("Advanced composite backend created with 3 memory type routes!")
```

```python
# Create an agent that understands the three memory types
memory_agent = create_deep_agent(
    model=model,
    system_prompt="""You are a helpful assistant with structured long-term memory.

Your memory is organized into three types:
- /memories/semantic/   -> Facts & knowledge (user preferences, project details)
- /memories/episodic/   -> Past experiences (session logs, interaction summaries)  
- /memories/procedural/ -> Instructions & rules (how to format reports, coding standards)

When asked to remember something, save it to the appropriate memory type.
Regular files (not in /memories/) are ephemeral and disappear after the conversation.

When asked what you remember or know about the user, ALWAYS use ls and read_file to check
the memory directories first. Do not rely on conversation context alone.

When referencing file paths, use backtick formatting like `path/file.md` instead of markdown links.
""",
    checkpointer=checkpointer,
    backend=advanced_composite_backend,
    store=memory_store,
)

# Thread 1: Write to all three memory types
config_t1 = {"configurable": {"thread_id": str(uuid7())}}

result = memory_agent.invoke({
    "messages": [{"role": "user", "content": """Please save the following to the appropriate memory types:
1. I prefer Python over JavaScript (this is a fact about me)
2. In our last session, we researched LangGraph and found it useful (this is a past experience)
3. Always use inline citations [1], [2] in research reports (this is a rule)"""}]
}, config=config_t1)

print(result["messages"][-1].content)
```

```python
# Thread 2: Verify memories persist across threads
config_t2 = {"configurable": {"thread_id": str(uuid7())}}

result = memory_agent.invoke({
    "messages": [{"role": "user", "content": "What do you remember about me? Check all memory types: semantic, episodic, and procedural."}]
}, config=config_t2)

print("From a NEW thread:\n")
print(result["messages"][-1].content)
```

### Namespace Scoping: Per-User vs Global Memory

By default, `StoreBackend` stores files in the namespace `(assistant_id, "filesystem")` -- meaning all users of one assistant share the same memories. But what if you want **per-user isolation**?

`StoreBackend` accepts a `namespace` parameter -- a callable that receives a `BackendContext` and returns a namespace tuple. This controls data isolation:

| Scope | Namespace | Who Can See It |
|-------|-----------|----------------|
| **Default** | `(assistant_id, "filesystem")` | All users of one assistant |
| **Per-user** | `("user", user_id, "filesystem")` | Only that specific user |
| **Global** | `("shared", "filesystem")` | All users across all assistants |

This lets you build agents where:
- `/memories/user/` stores private per-user preferences (isolated by `user_id`)
- `/memories/shared/` stores team guidelines visible to everyone

```python
scoped_store = InMemoryStore()

scoped_backend = CompositeBackend(
    default=StateBackend(),
    routes={
        # Per-user memories: namespace lambda receives the runtime,
        # so rt.context picks up the current invocation's user_id
        "/memories/user/": StoreBackend(
            namespace=lambda rt: (
                "user",
                getattr(rt.context, "user_id", "default"),
                "filesystem",
            ),
        ),
        # Shared memories: same namespace for ALL users
        "/memories/shared/": StoreBackend(
            namespace=lambda rt: ("shared", "filesystem"),
        ),
    },
)



scoped_agent = create_deep_agent(
    model=model,
    system_prompt="""You are a helpful assistant with scoped memory.

MEMORY SCOPES:
- /memories/user/    -> Private to the current user (only they can see it)
- /memories/shared/  -> Shared across all users (everyone can see it)

Save personal preferences to /memories/user/ and team guidelines to /memories/shared/.

When asked what you remember, ALWAYS use ls and read_file to check memory directories first.

When referencing file paths, use backtick formatting like `path/file.md` instead of markdown links.
""",
    checkpointer=checkpointer,
    backend=scoped_backend,
    store=scoped_store,
)

print("Scoped agent created with per-user and shared memory!")
```

```python
# User A writes both private and shared memories
config_alice = {"configurable": {"thread_id": str(uuid7()), "user_id": "alice"}}

result = scoped_agent.invoke({
    "messages": [{"role": "user", "content": """Save these two things:
1. To my private memory (/memories/user/): 'Alice prefers dark mode and Python'
2. To shared memory (/memories/shared/): 'Team guideline: Always write unit tests'"""}]
}, config=config_alice)

print("Alice wrote:\n", result["messages"][-1].content)
```

```python
# User B tries to read both - can they see Alice's private memories?
config_bob = {"configurable": {"thread_id": str(uuid7()), "user_id": "bob"}}

result = scoped_agent.invoke({
    "messages": [{"role": "user", "content": "List all files in `/memories/user/` and `/memories/shared/` to see what you can access."}]
}, config=config_bob)

print("Bob sees:\n", result["messages"][-1].content)
print("\n" + "=" * 50)
print("Bob can see shared guidelines but NOT Alice's private preferences!")
```

### Key Takeaway:
- **Three memory types** (semantic, episodic, procedural) map naturally to filesystem paths via `CompositeBackend` routes
- **`namespace`** on `StoreBackend` controls data isolation -- per-user, per-assistant, or global
- Longer route prefixes take precedence in `CompositeBackend` (e.g., `/memories/semantic/` over `/memories/`)
- Pass `user_id` via config to scope memories per user: `config={"configurable": {"user_id": "alice"}}`

## Part 8: AGENTS.md & Skills

So far we've been writing `system_prompt` strings directly in code. Deep Agents provides two file-based alternatives that are more powerful and follow best practices:

### AGENTS.md: Persistent Identity & Instructions

`AGENTS.md` files provide persistent context that is **always loaded** into the system prompt via the `memory=` parameter. This is where you put your agent's identity, workflow rules, and preferences. Another key benefit of AGENTS.md files, is that they're in simple markdown, which is easy for anyone to edit. 

The powerful part: **the agent can read AND edit its own AGENTS.md file**. This means the agent can update its own instructions based on feedback, essentially a self-modifiable system prompt.

> **Tradeoff**: AGENTS.md is powerful for development and internal agents -- the agent can learn and self-improve by editing its own instructions. But in **production**, you'd want to move critical instructions to `system_prompt` (which the agent can't edit) to prevent the agent from accidentally breaking its own behavior. Think of AGENTS.md as great for the "meta learning" phase.

### Skills: On-Demand Capabilities

Skills are reusable capabilities bundled as `SKILL.md` files with frontmatter metadata. They use **progressive disclosure**:
1. At startup, only the skill **name + description** (frontmatter) is loaded -- just a few tokens
2. When the agent determines a skill is relevant to the current task, it reads the **full SKILL.md** content
3. This keeps the prompt clean until the skill is actually needed

### Comparison

| Approach | Loaded When | Editable by Agent | Best For |
|----------|-------------|-------------------|----------|
| `system_prompt` | Always | No | Core identity, immutable rules |
| `AGENTS.md` (memory) | Always | Yes | Workflow, preferences, learnable rules |
| `SKILL.md` (skills) | On demand | No (read-only) | Task-specific instructions, templates |

```python
# Define our AGENTS.md -- the agent's identity and instructions
agents_md_content = """# Research Assistant

You are an expert research assistant that can search the web, synthesize findings and produce polished reports and content.

## Workflow

1. **Plan** -- Use `write_todos` to break the task into steps
2. **Research** -- Search for information using tavily_search
3. **Reflect** -- after each search reflect and analyze findings
4. **Synthesize** -- Combine findings into a comprehensive report
5. **Write** -- Save the final report to `/final_report.md`
6. **Remember** -- Save key takeaways to `/memories/research_notes.md`

## Rules

- Use 2-3 searches maximum
- Consolidate citations -- each unique URL gets one number [1], [2], [3]
- End reports with a Sources section
- Check for relevant skills when asked to create specific content formats
"""

print("AGENTS.md defined!")
print(agents_md_content)
```

### Now let's define our agent's Skills

```python
# Define LinkedIn post skill
linkedin_skill_content = """---
name: linkedin-post
description: Write a LinkedIn post based on research findings or a given topic. Use this skill when asked to create LinkedIn content, professional posts, or thought leadership pieces.
---

# LinkedIn Post Skill

## Format
- **Hook**: Start with a bold opening line that grabs attention (appears before the 'see more' cut)
- **Body**: 3-5 short paragraphs, each 1-2 sentences
- Use line breaks between paragraphs for readability
- Include 1-2 relevant emojis per paragraph (don't overdo it)
- End with a call-to-action or question to drive engagement
- Add 3-5 relevant hashtags at the bottom

## Tone
- Professional but conversational
- Share insights, not just information
- Use 'I' statements and personal perspective where appropriate

## Length
- Ideal: 150-300 words
- LinkedIn truncates after ~210 characters, so the first line must hook the reader
"""

print("LinkedIn post skill defined!")
print(f"Skill name: linkedin-post")
print(f"Length: {len(linkedin_skill_content)} chars")
```

```python
# Define Twitter/X post skill
twitter_skill_content = """---
name: twitter-post
description: Write a Twitter/X post or thread based on research findings or a given topic. Use this skill when asked to create tweets, X posts, or social media threads.
---

# Twitter/X Post Skill

## Single Tweet Format
- Maximum 280 characters
- Lead with the most compelling point
- Use numbers or data when possible
- 1-2 hashtags max (optional)

## Thread Format (for longer content)
- Tweet 1: Hook + preview (e.g., 'A thread on X:')
- Tweets 2-N: One idea per tweet, numbered (1/, 2/, 3/)
- Final tweet: Summary + call-to-action
- 4-8 tweets is the sweet spot

## Tone
- Concise and punchy
- Opinionated takes perform better than neutral summaries
- Use plain language -- no corporate speak
"""

print("Twitter/X post skill defined!")
print(f"Skill name: twitter-post")
print(f"Length: {len(twitter_skill_content)} chars")
```

### Put It All Together (AGENTS.md + Skills)

```python
# Create an agent that uses AGENTS.md + skills
# In a notebook, we seed the files via the `files` state key using create_file_data()
# In production (langgraph dev / Studio), you'd use FilesystemBackend and load from disk
from deepagents.backends.utils import create_file_data

skill_agent = create_deep_agent(
    model=model,
    tools=[tavily_search],
    system_prompt="You are an expert research assistant.",
    memory=["/AGENTS.md"],         # Always loaded into system prompt
    skills=["/skills/"],            # Loaded on demand via progressive disclosure
    checkpointer=checkpointer,
    backend=composite_backend,
    store=store,
)

# Seed the AGENTS.md and skill files into the virtual filesystem
skill_files = {
    "/AGENTS.md": create_file_data(agents_md_content),
    "/skills/linkedin-post/SKILL.md": create_file_data(linkedin_skill_content),
    "/skills/twitter-post/SKILL.md": create_file_data(twitter_skill_content),
}

print("Agent created with AGENTS.md + 2 skills!")
print(f"  Memory: `/AGENTS.md` (always loaded)")
print(f"  Skills: linkedin-post, twitter-post (loaded on demand)")
```

```python
# Test: Research a topic, then ask for a LinkedIn post
# The agent will research first (no skill needed), then load the linkedin-post skill on demand
config = {"configurable": {"thread_id": str(uuid7())}}

result = skill_agent.invoke(
    {
        "messages": [{"role": "user", "content": "Research what AI agents are, then write a LinkedIn post about your findings."}],
        "files": skill_files
    },
    config=config
)

print(result["messages"][-1].content)
```

```python
# Test: Now ask for a tweet thread in the same conversation
# The twitter-post skill will be loaded on demand
result = skill_agent.invoke(
    {
        "messages": [{"role": "user", "content": "Now write a Twitter/X thread about the same topic."}]
    },
    config=config
)

print(result["messages"][-1].content)
```

### Key Takeaway:
- **AGENTS.md** replaces hardcoded `system_prompt` strings -- loaded via `memory=`, editable by the agent
- **Skills** provide on-demand capabilities via progressive disclosure -- only loaded when relevant
- In a notebook, seed files via `files=` with `create_file_data()`; in production, use `FilesystemBackend` to load from disk
- The agent can self-improve by editing its own AGENTS.md -- powerful for development, but lock down critical rules in `system_prompt` for production

## Part 9: The Complete Research Agent

Let's review what we built! Starting from a basic `create_deep_agent()` call, we progressively added:

```
Part 1: create_deep_agent(model)                    → Basic filesystem agent
Part 2: + tools=[tavily_search]                     → Can search web
Part 3: (understand backends)                       → Same agent, understood storage
Part 4: + subagents=[research_agent]                → Can delegate research
Part 5: + middleware=[log_tool_calls]               → Log tool calls as they happen
Part 6: + interrupt_on={...}, checkpointer          → Human oversight
Part 7: + backend (CompositeBackend)                → Long-term memory
Part 8: + memory (AGENTS.md) + skills (SKILL.md)    → Self-modifiable identity + on-demand capabilities
```

Now let's build the final agent using the AGENTS.md + skills pattern:

```python
# Create our final research agent -- using AGENTS.md + skills (just like the Studio agent)
final_agent = create_deep_agent(
    model=model,
    tools=[tavily_search],
    system_prompt="You are an expert research assistant.",
    middleware=[log_tool_calls],
    skills=["/skills/"],            # LinkedIn + Twitter skills loaded on demand
    memory=["/AGENTS.md"],         # Identity + workflow from AGENTS.md
    checkpointer=checkpointer,
    backend=composite_backend,
    store=store,
)

# Seed the files into the virtual filesystem
final_agent_files = {
    "/AGENTS.md": create_file_data(agents_md_content),
    "/skills/linkedin-post/SKILL.md": create_file_data(linkedin_skill_content),
    "/skills/twitter-post/SKILL.md": create_file_data(twitter_skill_content),
}

print("Final research agent created with AGENTS.md + skills!")
```

```python
# Run a full research workflow -- research + LinkedIn post (exercises the skill!)
config = {"configurable": {"thread_id": str(uuid7())}}

print("Starting research workflow...\n")

result = final_agent.invoke(
    {
        "messages": [{"role": "user", "content": "Research what LangChain Deep Agents are, write a brief report, and then write a LinkedIn post about your findings."}],
        "files": final_agent_files
    },
    config=config,
)

print(result["messages"][-1].content[:2000])
```

```python
# Show the virtual filesystem - did the agent write files?
print("\n" + "=" * 60)
print("📁 VIRTUAL FILESYSTEM")
print("=" * 60)
for path, file_data in result.get("files", {}).items():
    if isinstance(file_data, dict) and "content" in file_data:
        content = "\n".join(file_data["content"])
    else:
        content = str(file_data)
    print(f"\n📄 '{path}' ({len(content)} chars)")
    print("-" * 40)
    print(content[:500] + ("..." if len(content) > 500 else ""))
```

### Key Takeaway:
- Deep Agents enables building complex agents incrementally
- Each capability (tools, subagents, HITL, memory) is composable
- The framework handles orchestration, you focus on capabilities

## Part 10: Next Steps

Congratulations! You've built a complete research agent using Deep Agents. Here's what you learned:

| Concept | What It Does |
|---------|-------------|
| **Agent Harness** | Pre-built tools + context management |
| **Custom Tools** | Extend capabilities (search) |
| **Backends** | Control file storage -- ephemeral, disk, persistent, or hybrid |
| **Subagents** | Context isolation for complex tasks |
| **Middleware** | Pluggable capability modules (filesystem, summarization, todos, etc.) |
| **Human-in-the-Loop** | Safety gates for sensitive operations |
| **Long-Term Memory** | Persistent storage with path routing + namespace scoping |
| **AGENTS.md** | File-based agent identity -- always loaded, editable by the agent |
| **Skills** | On-demand capabilities via progressive disclosure (SKILL.md files) |

### Resources

- [Deep Agents Documentation](https://docs.langchain.com/oss/python/deepagents/)
- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/)
- [LangChain Academy](https://academy.langchain.com/)
- [LangChain vs LangGraph vs Deep Agents](https://docs.langchain.com/oss/python/concepts/products)

### What's Next?

1. **Run in Studio** - Check out `agents/deep_agent/` for a production-ready version with AGENTS.md and skills on disk
2. **Deploy** - Use LangSmith Deployment for production (`langgraph dev` or LangSmith Deployments)
3. **Add Skills** - Create your own SKILL.md files for domain-specific capabilities
4. **Customize Memory** - Use namespace scoping for per-user vs shared memory
5. **Extend** - Build multi-agent systems with specialized subagents

<br>
<br>
---
<br>

**Happy building!**
