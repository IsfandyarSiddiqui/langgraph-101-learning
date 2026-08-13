# Deep Agents Extra: Under the Hood

The 201 taught you how to **use** Deep Agents. This notebook teaches you how they **work** — and how to customize the parts that matter.

**What you'll learn:**
- How `create_deep_agent()` is really just `create_agent()` + middleware
- How to build your own filesystem interface (the pattern, not the full implementation)
- How to replace the default `task()` tool with structured task definitions
- How to plug a deterministic `StateGraph` workflow in as a subagent
- How MCP adapters make external tools interchangeable with native ones
- How to use sandboxes for code execution
- What happens when context management fails — and how to fix it

**Prerequisites:** Complete the 201 Deep Agents notebook first.

## Part 0: Setup

```python
# Add project root to Python path
import sys
from pathlib import Path
project_root = Path().resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.models import model
print(f"Model: {model.__class__.__name__}")
```

```python
# Shared imports for the notebook
from langchain_core.tools import tool
from langsmith import uuid7
from langgraph.checkpoint.memory import MemorySaver

from tavily import TavilyClient
tavily_client = TavilyClient()

@tool(parse_docstring=True)
def tavily_search(query: str) -> str:
    """Search the web for information on a given query.
    
    Args:
        query: Search query to execute.
    """
    results = tavily_client.search(query, max_results=3)
    return "\n\n".join(
        f"**{r['title']}**\n{r['url']}\n{r['content']}" 
        for r in results["results"]
    )
```

---
## Part 1: Deconstructing `create_deep_agent`

`create_deep_agent()` is a wrapper around `create_agent()` that assembles a specific middleware stack.

Here's what it does (simplified):

```python
def create_deep_agent(model, tools, system_prompt, ...):
    middleware = [
        TodoListMiddleware(),                    # Planning
        FilesystemMiddleware(backend=backend),   # File tools
        SubAgentMiddleware(subagents=...),       # task() tool
        SummarizationMiddleware(model, ...),     # Context management
        PatchToolCallsMiddleware(),              # Fix dangling tool calls
    ]
    return create_agent(model, tools=tools, middleware=middleware)
```

### Why this matters

If you can build the middleware stack yourself, you can:
- Swap out any piece (e.g., your own filesystem, your own task tool)
- Add middleware in different order
- Use `create_agent()` directly without the deep agents dependency

```python
# Let's build the same thing from scratch using create_agent
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.summarization import create_summarization_middleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.backends import StateBackend

# This is the core of what create_deep_agent does
hand_built_agent = create_agent(
    model,
    tools=[tavily_search],
    system_prompt="You are a helpful research assistant.",
    middleware=[
        TodoListMiddleware(),
        FilesystemMiddleware(backend=StateBackend),
        create_summarization_middleware(model, StateBackend),
        PatchToolCallsMiddleware(),
    ],
).with_config({"recursion_limit": 1000})

print("Agent built with create_agent + middleware")
print(f"Type: {type(hand_built_agent).__name__}")
```

```python
# Test it -- it behaves exactly like create_deep_agent
config = {"configurable": {"thread_id": str(uuid7())}}

result = hand_built_agent.invoke({
    "messages": [{"role": "user", "content": "Write a file called /notes.md with 'Built from scratch!' then read it back."}]
}, config=config)

print(result["messages"][-1].content)
print("\nFiles in state:", list(result.get("files", {}).keys()))
```

### Middleware hooks at a glance

Each middleware can implement these hooks:

| Hook | When it runs | Common use |
|------|-------------|------------|
| `before_agent(state, runtime)` | Once at start | Load skills, memory |
| `wrap_model_call(request, handler)` | Each LLM call | Inject system prompt text |
| `wrap_tool_call(request, handler)` | Each tool call | Log, intercept, modify |
| `tools` (property) | Registration time | Add tools to the agent |
| `state_schema` (property) | Graph compilation | Extend state with new fields |

Let's see these hooks in action with a simple logging middleware:

```python
from langchain.agents.middleware import wrap_model_call, wrap_tool_call

@wrap_model_call
def inspect_model_call(request, handler):
    """See what the LLM receives on each call."""
    # request.system_message contains the full system prompt (all middleware have appended to it)
    sys_msg = request.system_message
    if sys_msg:
        # System message can be multi-block from multiple middleware
        total_chars = sum(len(b.get("text", "")) for b in sys_msg.content_blocks if isinstance(b, dict))
        print(f"\n🧠 [Model Call] System prompt: {total_chars:,} chars across {len(sys_msg.content_blocks)} blocks")
    print(f"   Tools available: {[t.name for t in request.tools]}")
    print(f"   Messages in context: {len(request.messages)}")
    return handler(request)

@wrap_tool_call
def inspect_tool_call(request, handler):
    """See what tools are called and with what args."""
    name = request.tool_call["name"]
    args = request.tool_call["args"]
    # Truncate long args for display
    display_args = {k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v) for k, v in args.items()}
    print(f"   🔧 {name}({display_args})")
    return handler(request)

# Add our inspect middleware to the stack
inspectable_agent = create_agent(
    model,
    tools=[tavily_search],
    system_prompt="You are a helpful assistant.",
    middleware=[
        TodoListMiddleware(),
        FilesystemMiddleware(backend=StateBackend),
        create_summarization_middleware(model, StateBackend),
        PatchToolCallsMiddleware(),
        inspect_model_call,  # Our custom middleware
        inspect_tool_call,
    ],
).with_config({"recursion_limit": 1000})

config = {"configurable": {"thread_id": str(uuid7())}}
result = inspectable_agent.invoke({
    "messages": [{"role": "user", "content": "Write a file called /hello.txt with 'hi'"}]
}, config=config)

print("\n---")
print(result["messages"][-1].content)
```

### Key Takeaway
- `create_deep_agent()` = `create_agent()` + a specific middleware stack
- Middleware hooks let you inspect, modify, and extend any part of the agent loop
- You can swap, reorder, or replace any middleware to customize behavior

---
## Part 2: Building a Minimal Filesystem Interface

Deep Agent's `FilesystemMiddleware` provides 7 tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`). Rather than reimplement all of them, let's build a minimal 2-tool version to understand the **pattern**.

**Filesystem tools don't touch the filesystem directly**. They:
1. Operate through a backend abstraction (`BackendProtocol`)
2. Return `Command(update={"files": {...}})` to update LangGraph state
3. Use a custom state reducer to merge file updates

This pattern separates *where* files are stored from *how* the agent uses them.

```python
from datetime import datetime, timezone
from typing import Annotated, Any, NotRequired
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from langchain_core.messages import SystemMessage


# Step 1: Define the state reducer
# This is how LangGraph merges file updates from multiple tool calls
def file_reducer(left: dict | None, right: dict) -> dict:
    """Merge file state updates. None values = deletions."""
    result = {**(left or {})}
    for key, value in right.items():
        if value is None:
            result.pop(key, None)  # Delete
        else:
            result[key] = value    # Create/update
    return result


# Step 2: Define the state schema
class MinimalFilesystemState(AgentState):
    files: Annotated[NotRequired[dict], file_reducer]


# Step 3: Build the middleware
class MinimalFilesystemMiddleware(AgentMiddleware):
    """A stripped-down filesystem middleware to show the pattern."""
    
    state_schema = MinimalFilesystemState
    
    def __init__(self):
        super().__init__()
        self.tools = [self._make_write_tool(), self._make_read_tool()]
    
    def _make_write_tool(self):
        def write_file(
            file_path: Annotated[str, "Absolute path for the file (e.g., /notes.md)"],
            content: Annotated[str, "Content to write"],
            runtime: ToolRuntime,
        ) -> Command:
            """Write a file to the virtual filesystem."""
            now = datetime.now(timezone.utc).isoformat()
            file_data = {
                "content": content.split("\n"),
                "created_at": now,
                "modified_at": now,
            }
            # THIS is the key pattern: return a Command to update state
            return Command(update={
                "files": {file_path: file_data},
                "messages": [ToolMessage(
                    f"Wrote {len(content)} chars to {file_path}",
                    tool_call_id=runtime.tool_call_id,
                )],
            })
        
        return StructuredTool.from_function(name="write_file", func=write_file)
    
    def _make_read_tool(self):
        def read_file(
            file_path: Annotated[str, "Absolute path of the file to read"],
            runtime: ToolRuntime,
        ) -> Command:
            """Read a file from the virtual filesystem."""
            files = runtime.state.get("files", {})
            if file_path not in files:
                return Command(update={
                    "messages": [ToolMessage(
                        f"Error: {file_path} not found",
                        tool_call_id=runtime.tool_call_id,
                    )],
                })
            content = "\n".join(files[file_path]["content"])
            return Command(update={
                "messages": [ToolMessage(
                    content,
                    tool_call_id=runtime.tool_call_id,
                )],
            })
        
        return StructuredTool.from_function(name="read_file", func=read_file)
    
    def wrap_model_call(self, request, handler):
        """Tell the LLM about its filesystem capabilities."""
        instructions = """## Filesystem
You have `write_file` and `read_file` tools. Use absolute paths (e.g., `/notes.md`).
Files are stored in-memory and persist within this thread."""
        # Append our instructions to the existing system message
        existing = request.system_message
        blocks = list(existing.content_blocks) if existing else []
        blocks.append({"type": "text", "text": f"\n\n{instructions}"})
        new_sys = SystemMessage(content_blocks=blocks)
        return handler(request.override(system_message=new_sys))


print("MinimalFilesystemMiddleware defined")
print("Tools:", [t.name for t in MinimalFilesystemMiddleware().tools])
```

```python
# Use our minimal filesystem with create_agent
minimal_fs_agent = create_agent(
    model,
    system_prompt="You are a helpful assistant.",
    middleware=[MinimalFilesystemMiddleware()],
)

config = {"configurable": {"thread_id": str(uuid7())}}
result = minimal_fs_agent.invoke({
    "messages": [{"role": "user", "content": "Write a file /demo.txt with 'Hello from minimal filesystem!' then read it back."}]
}, config=config)

print("Agent response:", result["messages"][-1].content)
print("\nFiles in state:", {k: v["content"] for k, v in result.get("files", {}).items()})
```

### How this compares to the real `FilesystemMiddleware`

Our minimal version shows the core pattern. The real implementation adds:

| Feature | Our version | Real `FilesystemMiddleware` |
|---------|-------------|----------------------------|
| Tools | `write_file`, `read_file` | + `ls`, `edit_file`, `glob`, `grep`, `execute` |
| Storage | Direct state access | `BackendProtocol` abstraction |
| Pagination | No | `offset`/`limit` on reads |
| Images | No | Multimodal content blocks |
| Sandboxing | No | `SandboxBackendProtocol` |
| Path validation | No | Sanitization, virtual paths |

### Key Takeaway
- Filesystem tools modify state via `Command(update={"files": {...}})`, not by writing to disk
- A custom state reducer (`file_reducer`) handles merging and deletions
- `wrap_model_call` injects instructions so the LLM knows what tools are available
- The real implementation adds a backend abstraction so storage is pluggable

---
## Part 3: Backend Implementations — Where Files Actually Live

In Part 2 we built a filesystem that stores files in LangGraph state. But that's just one option. Deep Agents has three backend implementations, each with fundamentally different storage semantics.

**tools don't know where files are stored**. They call `backend.write()` and the backend decides whether to update state, write to disk, or call a remote store.

### The `WriteResult` branching pattern

Every backend's `write()` returns a `WriteResult` with three fields:
```python
@dataclass
class WriteResult:
    error: str | None = None        # Error message on failure
    path: str | None = None          # Path of written file
    files_update: dict | None = None  # State update dict (or None)
```

The `files_update` field is what controls the branching:
- **`StateBackend`**: Returns `files_update={path: file_data}` → middleware wraps this in `Command(update={"files": ...})`
- **`FilesystemBackend`**: Returns `files_update=None` → file is already on disk, middleware just returns a string confirmation
- **`StoreBackend`**: Returns `files_update=None` → file is already in the Store, no state update needed

`FilesystemMiddleware._create_write_file_tool()`:
```python
res: WriteResult = backend.write(file_path, content)
if res.files_update is not None:
    # StateBackend: update LangGraph state
    return Command(update={"files": res.files_update, "messages": [...]})
# FilesystemBackend/StoreBackend: already persisted
return f"Updated file {res.path}"
```

```python
# Let's see the difference in practice
from deepagents.backends.protocol import WriteResult

# Simulate what each backend returns for the same write() call

# StateBackend: "here's the data, put it in state"
state_result = WriteResult(
    path="/notes.md",
    files_update={"/notes.md": {"content": ["hello"], "created_at": "...", "modified_at": "..."}}
)
print(f"StateBackend:      files_update = {type(state_result.files_update).__name__} (middleware issues Command)")

# FilesystemBackend: "I already wrote it to disk"
fs_result = WriteResult(path="/notes.md", files_update=None)
print(f"FilesystemBackend: files_update = {fs_result.files_update} (middleware returns string)")

# StoreBackend: "I already put it in the Store" 
store_result = WriteResult(path="/notes.md", files_update=None)
print(f"StoreBackend:      files_update = {store_result.files_update} (middleware returns string)")
```

### StateBackend — Ephemeral, in LangGraph state

This is the default. Files live in `runtime.state["files"]` — a dict managed by the LangGraph checkpointer.

```python
class StateBackend(BackendProtocol):
    def __init__(self, runtime: ToolRuntime):
        self.runtime = runtime
    
    def read(self, file_path, ...):
        files = self.runtime.state.get("files", {})
        return files[file_path]  # Read directly from state
    
    def write(self, file_path, content):
        file_data = create_file_data(content)
        # Return the data so middleware can update state via Command
        return WriteResult(path=file_path, files_update={file_path: file_data})
```

**Lifecycle:** Files persist within a thread (via checkpointer) but vanish across threads.

**Trade-off:** Fast (in-memory), but every file is serialized into the checkpoint. Large files bloat checkpoint size.

```python
# StateBackend in action — files live in state["files"]
from deepagents import create_deep_agent
from deepagents.backends import StateBackend

state_agent = create_deep_agent(
    model=model,
    backend=StateBackend(),  # Default — pass an instance
    system_prompt="You are a helpful assistant.",
)

config = {"configurable": {"thread_id": str(uuid7())}}
result = state_agent.invoke({
    "messages": [{"role": "user", "content": "Write a file /demo.txt with 'hello from StateBackend'"}]
}, config=config)

# Files are in the result state
print("Files in state:", list(result.get("files", {}).keys()))
if "/demo.txt" in result.get("files", {}):
    print("Content:", result["files"]["/demo.txt"]["content"])
```

### FilesystemBackend — Real disk I/O

Reads and writes actual files on disk. The key parameter is `virtual_mode`:

- `virtual_mode=True`: Paths are sandboxed under `root_dir`. Agent writes to `/notes.md` → actually writes to `{root_dir}/notes.md`. Path traversal (`..`) is blocked.
- `virtual_mode=False` (default, deprecated): Absolute paths work as-is. **No security boundary.**

```python
class FilesystemBackend(BackendProtocol):
    def __init__(self, root_dir, virtual_mode=True):
        self.cwd = Path(root_dir).resolve()
        self.virtual_mode = virtual_mode
    
    def write(self, file_path, content):
        resolved = self._resolve_path(file_path)  # Sandboxing happens here
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        return WriteResult(path=file_path, files_update=None)  # Already on disk!
```

**Key difference from StateBackend:** `files_update=None` — the file is already persisted to disk. The middleware returns a plain string response, no `Command` needed.

**Lifecycle:** Files persist on disk permanently. No checkpointer needed for persistence, but files are shared across all threads that use the same `root_dir`.

```python
# FilesystemBackend in action — files go to real disk
import tempfile, os
from deepagents.backends import FilesystemBackend

sandbox_dir = tempfile.mkdtemp(prefix="deepagents_extra_")
print(f"Sandbox directory: {sandbox_dir}")

fs_agent = create_deep_agent(
    model=model,
    backend=FilesystemBackend(root_dir=sandbox_dir, virtual_mode=True),
    system_prompt="You are a helpful assistant.",
)

config = {"configurable": {"thread_id": str(uuid7())}}
result = fs_agent.invoke({
    "messages": [{"role": "user", "content": "Write a file /demo.txt with 'hello from FilesystemBackend'"}]
}, config=config)

# No files in state — they're on disk!
print(f"Files in state: {list(result.get('files', {}).keys())}")

# But they exist on disk
disk_path = os.path.join(sandbox_dir, "demo.txt")
if os.path.exists(disk_path):
    print(f"File on disk: {open(disk_path).read()}")

# Cleanup
import shutil
shutil.rmtree(sandbox_dir, ignore_errors=True)
```

### StoreBackend — Persistent, cross-thread storage

Uses LangGraph's `BaseStore` (a key-value store). Files persist across all threads — write in thread A, read in thread B.

```python
class StoreBackend(BackendProtocol):
    def __init__(self, runtime: ToolRuntime, *, namespace=None):
        self.runtime = runtime
    
    def write(self, file_path, content):
        store = self.runtime.store          # LangGraph provides this
        namespace = self._get_namespace()    # e.g., ("filesystem",)
        file_data = create_file_data(content)
        store.put(namespace, file_path, file_data)  # Persisted immediately
        return WriteResult(path=file_path, files_update=None)  # Already in store!
```

**Namespace scoping** controls isolation:
- `("filesystem",)` — all threads/users share files
- `(user_id, "filesystem")` — per-user isolation
- `(assistant_id, "filesystem")` — per-assistant isolation

**Lifecycle:** Files persist as long as the Store exists. In LangGraph Platform, this means across deployments.

```python
# StoreBackend in action — files persist across threads
from langgraph.store.memory import InMemoryStore
from deepagents.backends import StoreBackend

store = InMemoryStore()

store_agent = create_deep_agent(
    model=model,
    backend=StoreBackend(namespace=lambda rt: ("filesystem",)),
    store=store,
    system_prompt="You are a helpful assistant.",
    checkpointer=MemorySaver(),
)

# Thread 1: Write a file
config_t1 = {"configurable": {"thread_id": str(uuid7())}}
result = store_agent.invoke({
    "messages": [{"role": "user", "content": "Write a file /shared_note.txt with 'This persists across threads!'"}]
}, config=config_t1)
print("Thread 1:", result["messages"][-1].content)

# Thread 2: Read it (different thread!)
config_t2 = {"configurable": {"thread_id": str(uuid7())}}
result = store_agent.invoke({
    "messages": [{"role": "user", "content": "Read the file /shared_note.txt"}]
}, config=config_t2)
print("Thread 2:", result["messages"][-1].content)
```

### Backend comparison

| | StateBackend | FilesystemBackend | StoreBackend |
|---|---|---|---|
| **Storage** | LangGraph state (checkpoint) | Real filesystem | LangGraph Store (key-value) |
| **Persistence** | Within thread | Permanent on disk | Cross-thread, cross-session |
| **`files_update`** | `{path: data}` → `Command` | `None` → string | `None` → string |
| **State bloat** | Yes (files in checkpoint) | No | No |
| **Use case** | Scratch space, notebooks | CLI tools, local dev | LangGraph Platform, production |
| **Needs `store=`** | No | No | Yes (`InMemoryStore` or platform store) |

In production, you typically use `CompositeBackend` to mix them — ephemeral scratch space via `StateBackend`, persistent memories via `StoreBackend`:

```python
composite_backend = CompositeBackend(
    default=StateBackend(),           # Scratch files go to state
    routes={
        "/memories/": StoreBackend(),  # Memories persist in Store
    },
)
```

### Key Takeaway
- The `WriteResult.files_update` field controls whether the middleware issues a `Command` or returns a string
- `StateBackend` returns data for state updates; `FilesystemBackend` and `StoreBackend` persist directly and return `None`
- All three backends implement the same `BackendProtocol` — tools don't know which one they're using
- `CompositeBackend` routes different paths to different backends for hybrid storage

---
## Part 4: Custom Task Definitions for Subagents

By default, `SubAgentMiddleware` creates a `task()` tool with two parameters:
- `description` (str) — a raw string describing what to do
- `subagent_type` (str) — which subagent to invoke

The subagent receives `HumanMessage(content=description)` — just a flat string.

This works, but has limitations:
- No structure — the LLM has to pack everything into prose
- No type safety — no validation on what gets passed
- No selective context — the subagent gets all non-excluded state

Let's build custom task tools that fix these issues.

```python
# The default task tool flow (simplified):
# 1. Main agent calls: task(description="Research LangGraph", subagent_type="researcher")
# 2. SubAgentMiddleware:
#    a. Filters state: removes internal keys (messages, todos, etc.) so subagents get clean context
#    b. Sets messages: [HumanMessage("Research LangGraph")]  <-- just the raw string
#    c. Invokes subagent
#    d. Returns: Command(update={"messages": [ToolMessage(result)]})
#
# The limitation: "description" is a raw string. The LLM has to pack
# everything — topic, depth, format, constraints — into prose.
# Let's fix that with structured inputs.
```

```python
# Custom Task Definition 1: Structured Input
#
# Instead of a raw string, define a Pydantic model for the task input.
# This gives the orchestrator agent a clear schema to fill out.

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage


class ResearchTask(BaseModel):
    """A structured research task definition."""
    topic: str = Field(description="The topic to research")
    depth: str = Field(description="'shallow' (quick overview), 'medium' (key details), or 'deep' (comprehensive)")
    output_format: str = Field(description="'bullet_points', 'narrative', or 'structured_report'")
    max_sources: int = Field(default=3, description="Maximum number of sources to consult")
    focus_areas: list[str] = Field(default_factory=list, description="Specific aspects to focus on")


def build_structured_task_tool(subagent_graph):
    """Build a task tool with structured input instead of a raw string."""
    
    def research(
        topic: Annotated[str, "The topic to research"],
        depth: Annotated[str, "'shallow', 'medium', or 'deep'"],
        output_format: Annotated[str, "'bullet_points', 'narrative', or 'structured_report'"],
        max_sources: Annotated[int, "Maximum sources to consult"] = 3,
        focus_areas: Annotated[list[str], "Specific aspects to focus on"] = [],
        runtime: ToolRuntime = None,
    ) -> Command:
        """Delegate a structured research task to the research subagent."""
        # Build a well-structured prompt from the typed fields
        focus_str = "\n".join(f"  - {area}" for area in focus_areas) if focus_areas else "  (none specified)"
        
        prompt = f"""## Research Task

**Topic:** {topic}
**Depth:** {depth}
**Max Sources:** {max_sources}

**Focus Areas:**
{focus_str}

**Output Format:** {output_format}

Conduct the research and return your findings in the specified format."""
        
        # Start with clean state: just the structured prompt and shared files
        subagent_state = {
            "messages": [HumanMessage(content=prompt)],
        }
        # Optionally pass files if they exist in parent state
        files = runtime.state.get("files")
        if files:
            subagent_state["files"] = files
        
        result = subagent_graph.invoke(subagent_state)
        
        return Command(update={
            "messages": [ToolMessage(
                result["messages"][-1].text.rstrip(),
                tool_call_id=runtime.tool_call_id,
            )],
        })
    
    return StructuredTool.from_function(
        name="research",
        func=research,
        description="Delegate a structured research task to a specialist subagent.",
    )


print("Structured task tool builder defined")
```

```python
# Create the research subagent and the orchestrator

# Subagent: a simple create_agent with search capabilities
research_subagent = create_agent(
    model,
    tools=[tavily_search],
    system_prompt="You are a research specialist. Follow the task instructions precisely.",
    middleware=[
        PatchToolCallsMiddleware(),
    ],
)

# Build the structured task tool
structured_research_tool = build_structured_task_tool(research_subagent)

# Orchestrator: uses the structured task tool
orchestrator = create_agent(
    model,
    tools=[structured_research_tool],
    system_prompt="""You are a research coordinator. When asked to research something:
1. Use the `research` tool to delegate to a specialist
2. Always specify depth, output_format, and relevant focus_areas
3. Summarize the results for the user""",
    middleware=[
        TodoListMiddleware(),
        FilesystemMiddleware(backend=StateBackend),
        PatchToolCallsMiddleware(),
    ],
).with_config({"recursion_limit": 100})

print("Orchestrator ready with structured research tool")
```

```python
# Test: The orchestrator now passes structured fields, not raw strings
config = {"configurable": {"thread_id": str(uuid7())}}

result = orchestrator.invoke({
    "messages": [{"role": "user", "content": "Research what's new with LangGraph in 2025. Focus on the agent framework and middleware system. Give me bullet points."}]
}, config=config)

print(result["messages"][-1].content[:2000])
```

### Key Takeaway
- The default `task()` tool passes a raw string as `HumanMessage` — you can replace this with structured Pydantic fields
- Custom task tools give you full control over what state the subagent receives

---
## Part 5: StateGraph Workflows as Subagents

Subagents don't have to be agents (LLM loops). You can plug in any `Runnable` that has `messages` in its output — including a deterministic `StateGraph` workflow.

This is powerful when you have a **fixed pipeline** (gather → analyze → format) that shouldn't be left to an LLM to orchestrate.

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# Define the workflow state — must include `messages` for CompiledSubAgent
class ResearchWorkflowState(TypedDict):
    messages: list          # Required: CompiledSubAgent reads the last message
    topic: str              # Extracted from the incoming HumanMessage
    raw_data: str           # Output of the gather step
    analysis: str           # Output of the analyze step


# Step 1: Extract the topic and gather data
def gather_data(state: ResearchWorkflowState):
    """Search for information on the topic."""
    # The task tool sends HumanMessage(description) -- extract the topic
    topic = state["messages"][-1].content
    
    # Use tavily directly (not as a tool -- this is a deterministic step)
    results = tavily_client.search(topic, max_results=3)
    raw = "\n\n".join(
        f"Source: {r['url']}\n{r['content']}" 
        for r in results["results"]
    )
    return {"topic": topic, "raw_data": raw}


# Step 2: LLM analysis of the raw data
def analyze_data(state: ResearchWorkflowState):
    """Have the LLM analyze and synthesize the gathered data."""
    response = model.invoke([
        {"role": "system", "content": "Analyze the following research data. Identify key themes, facts, and insights. Be concise."},
        {"role": "user", "content": f"Topic: {state['topic']}\n\nData:\n{state['raw_data']}"}
    ])
    return {"analysis": response.content}


# Step 3: Format the final report and set messages for the parent agent
def format_report(state: ResearchWorkflowState):
    """Format into a final report and return via messages."""
    response = model.invoke([
        {"role": "system", "content": "Write a concise research summary with bullet points based on this analysis."},
        {"role": "user", "content": state["analysis"]}
    ])
    # The last message is what gets returned to the parent agent
    return {"messages": [AIMessage(content=response.content)]}


# Build the workflow graph
workflow = StateGraph(ResearchWorkflowState)
workflow.add_node("gather", gather_data)
workflow.add_node("analyze", analyze_data)
workflow.add_node("format", format_report)
workflow.add_edge(START, "gather")
workflow.add_edge("gather", "analyze")
workflow.add_edge("analyze", "format")
workflow.add_edge("format", END)

compiled_workflow = workflow.compile()
print("Research workflow compiled: gather → analyze → format")
```

```python
# Test the workflow standalone first
standalone_result = compiled_workflow.invoke({
    "messages": [HumanMessage(content="Latest developments in AI agents 2025")]
})

print("Workflow output (last message):")
print(standalone_result["messages"][-1].content[:1000])
```

```python
# Now plug it in as a CompiledSubAgent
from deepagents import create_deep_agent

agent_with_workflow = create_deep_agent(
    model=model,
    tools=[tavily_search],
    system_prompt="You are a research coordinator. Use the research-pipeline for structured research.",
    subagents=[
        {
            # CompiledSubAgent: any Runnable with 'messages' in state
            "name": "research-pipeline",
            "description": "A deterministic research pipeline that gathers data, analyzes it, and produces a structured report. Use this for any research request.",
            "runnable": compiled_workflow,
        },
    ],
)

config = {"configurable": {"thread_id": str(uuid7())}}
result = agent_with_workflow.invoke({
    "messages": [{"role": "user", "content": "Research the current state of LangGraph's middleware system"}]
}, config=config)

print(result["messages"][-1].content[:2000])
```

### Agent subagent vs. Workflow subagent

| | Agent subagent (default) | Workflow subagent |
|---|---|---|
| **Execution** | LLM decides steps | Fixed pipeline |
| **Flexibility** | Can adapt on the fly | Deterministic |
| **Cost** | More tokens (LLM reasoning) | Less tokens (structured steps) |
| **Debugging** | Harder (LLM choices vary) | Easier (fixed path) |
| **When to use** | Open-ended tasks | Well-defined pipelines |

The `CompiledSubAgent` contract is simple: your `Runnable` must have `messages` in its output state. The last message is returned to the parent as a `ToolMessage`.

### Key Takeaway
- Any `Runnable` with `messages` in state can be a subagent via `CompiledSubAgent`
- Deterministic workflows are cheaper and more predictable than agent subagents
- Mix both: use agent subagents for open-ended tasks, workflow subagents for pipelines

---
## Part 6: MCP Adapters for Tools

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) provides a standard way to expose tools from external servers. The key insight: **MCP tools become regular LangChain `BaseTool` objects**, so they're interchangeable with any other tool.

### How MCP tool adapters work

```
┌─────────────┐     ┌───────────────┐     ┌─────────────┐
│  MCP Server  │────▶│  MCP Adapter   │────▶│  BaseTool    │
│ (external)   │     │ (langchain-mcp)│     │ (standard)   │
└─────────────┘     └───────────────┘     └─────────────┘
                                                │
                    ┌───────────────────────────┘
                    ▼
        create_agent(tools=[mcp_tool, regular_tool])
```

1. **Write** an MCP server with `FastMCP` (or use an existing one)
2. **Connect** via stdio or HTTP/SSE using `langchain-mcp-adapters`
3. **Load** tools with `load_mcp_tools()` — they become standard `BaseTool` objects
4. **Use** them like any other LangChain tool

Let's build a real example using an MCP server with email tools.

```python
# Our MCP server lives at mcp/email_tools.py — a FastMCP server with 3 tools:
#   write_email, check_calendar_availability, schedule_meeting
# It runs via stdio transport (spawned as a subprocess).
#
# Let's connect to it and load the tools.

import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

mcp_server_path = Path().resolve().parent.parent / "mcp" / "email_tools.py"
server_params = StdioServerParameters(
    command=sys.executable,              # Same Python interpreter (respects venv)
    args=["-u", str(mcp_server_path)],   # -u = unbuffered stdout (required for stdio)
)

async def demo_mcp_tools():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # load_mcp_tools converts MCP tools → standard LangChain BaseTool objects
            tools = await load_mcp_tools(session)
            
            print(f"Loaded {len(tools)} MCP tools:\n")
            for tool in tools:
                print(f"  {tool.name}: {tool.description}")
                print(f"    Type: {type(tool).__name__}")
                print()
            
            # Call one directly to prove it works
            result = await tools[0].ainvoke({"to": "alice@example.com", "subject": "Hello", "content": "Hi Alice!"})
            print(f"Direct tool call result: {result}")

await demo_mcp_tools()
```

```python
# Use MCP tools with create_agent — they're standard BaseTool objects

async def demo_agent_with_mcp():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            
            # Mix MCP tools with regular tools
            all_tools = [tavily_search, *tools]
            
            agent = create_agent(
                model,
                tools=all_tools,
                system_prompt="You are a helpful assistant with email and calendar capabilities.",
                middleware=[PatchToolCallsMiddleware()],
            )
            
            config = {"configurable": {"thread_id": str(uuid7())}}
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": "Check my calendar availability for Monday and send an email to bob@example.com letting him know when I'm free."}]
            }, config=config)
            
            print(result["messages"][-1].content)

await demo_agent_with_mcp()
```

### When MCP changes your invocation pattern

MCP tools are **remote procedure calls**. This has practical implications:

| Aspect | Native `@tool` | MCP Tool |
|--------|----------------|----------|
| Latency | In-process | Network roundtrip (subprocess or HTTP) |
| State access | `ToolRuntime` available | No state access (stateless RPC) |
| Error handling | Python exceptions | MCP error protocol |
| Lifecycle | Lives with the agent | Server process must be running |

The most important difference: **MCP tools can't access `ToolRuntime`**. They can't read agent state, return `Command` objects, or modify state directly. They're pure input → output functions.

This means you can't replace `FilesystemMiddleware` tools with MCP filesystem tools and get the same state management. MCP tools are best for **external capabilities** (email, calendars, databases, APIs) rather than state-coupled operations.

### Key Takeaway
- MCP servers expose tools via `FastMCP` — write once, use from any MCP client
- `load_mcp_tools()` converts MCP tools into standard `BaseTool` objects
- Mix MCP and native tools freely — the agent doesn't know the difference
- MCP tools are stateless RPCs — they can't access agent state or return `Command` objects

---
## Part 7: Sandbox Usage

Deep Agents can execute shell commands when the backend implements `SandboxBackendProtocol`. This adds an `execute` tool to the agent.

### The sandbox backend hierarchy

```
BackendProtocol               <- File operations only (ls, read, write, edit, glob, grep)
  └── SandboxBackendProtocol  <- + execute(command, timeout) + id property
        └── BaseSandbox       <- Implements ALL file ops as shell commands via execute()
              └── LangSmithSandbox   <- Wraps langsmith.sandbox.SandboxClient
              └── YourOwnSandbox     <- Just implement execute()!
```

`BaseSandbox` implements **every** file operation (`read`, `write`, `edit`, `ls`, `glob`, `grep`) by shelling out commands through `execute()`. So to build your own sandbox backend, you only need to implement **one method**: `execute()`.

### LangSmith Sandbox SDK

```python
from langsmith.sandbox import SandboxClient

client = SandboxClient()  # Uses LANGSMITH_API_KEY
client.create_template(name="my-sandbox", image="python:3.12-slim")

with client.sandbox(template_name="my-sandbox") as sb:
    sb.write("/app/script.py", "print('hello')")
    result = sb.run("python /app/script.py")
    print(result.stdout)  # "hello\n"
```

Deep Agents wraps this with `LangSmithSandbox(sb)` to expose it as a `SandboxBackendProtocol`.

```python
# The protocol that gates sandbox capabilities
from deepagents.backends.protocol import BackendProtocol, SandboxBackendProtocol, ExecuteResponse

# BackendProtocol methods (file operations):
backend_methods = [m for m in dir(BackendProtocol) if not m.startswith('_') and not m.startswith('a')]
print("BackendProtocol methods:", backend_methods)

# SandboxBackendProtocol adds:
sandbox_only = [m for m in dir(SandboxBackendProtocol) if not m.startswith('_') and not m.startswith('a') and m not in dir(BackendProtocol)]
print("SandboxBackendProtocol adds:", sandbox_only)

print(f"\nExecuteResponse fields: output (str), exit_code (int|None), truncated (bool)")
```

```python
# (Optional) Running example: LangSmith Sandbox
# Requires LANGSMITH_API_KEY with a plan that supports sandboxes.
# Skip this cell if you don't have access — the conceptual section below covers the pattern.

import os

if os.environ.get("LANGSMITH_API_KEY"):
    try:
        from langsmith.sandbox import SandboxClient
        from deepagents.backends.langsmith import LangSmithSandbox

        client = SandboxClient()

        # Create a template (only needed once — safe to call again)
        try:
            client.create_template(name="deepagents-extra", image="python:3.12-slim")
            print("Created template 'deepagents-extra'")
        except Exception:
            print("Template 'deepagents-extra' already exists")

        # Spin up a sandbox and run commands
        with client.sandbox(template_name="deepagents-extra") as sb:
            # Direct SDK usage
            result = sb.run("echo 'Hello from sandbox!' && python3 --version")
            print(f"\nDirect execution:")
            print(f"  stdout: {result.stdout.strip()}")
            print(f"  success: {result.success}")

            # File operations
            sb.write("/app/hello.py", "print('Hello from a sandboxed Python script!')")
            result = sb.run("python3 /app/hello.py")
            print(f"\nScript execution:")
            print(f"  stdout: {result.stdout.strip()}")

            # Wrap as a Deep Agents backend and use with create_deep_agent
            backend = LangSmithSandbox(sb)
            print(f"\nSandbox ID: {backend.id}")
            print(f"Backend type: {type(backend).__name__}")
            print(f"Is SandboxBackendProtocol: {isinstance(backend, SandboxBackendProtocol)}")

            sandbox_agent = create_deep_agent(
                model=model,
                backend=backend,
                system_prompt="You are a coding assistant. Write and run code in the sandbox.",
            )

            config = {"configurable": {"thread_id": str(uuid7())}}
            result = sandbox_agent.invoke({
                "messages": [{"role": "user", "content": "Write a Python one-liner to /app/fib.py that prints the first 10 fibonacci numbers, then run it."}]
            }, config=config)
            print(f"\nAgent response: {result['messages'][-1].content}")

    except Exception as e:
        print(f"Sandbox error: {e}")
        print("Sandboxes require a LangSmith plan that supports them.")
else:
    print("LANGSMITH_API_KEY not set — skipping LangSmith sandbox demo.")
    print("Set it in your .env to run this cell.")
```

```python
# Building your own sandbox backend
#
# BaseSandbox handles all file operations via execute().
# You only need to implement: execute(), id, upload_files(), download_files()
#
# Here's the pattern (simplified from langchain-ai/openshell-deepagent):

from deepagents.backends.sandbox import BaseSandbox
from deepagents.backends.protocol import ExecuteResponse, FileUploadResponse, FileDownloadResponse
import subprocess

class LocalDockerSandbox(BaseSandbox):
    """Example: run commands in a local Docker container.
    
    In production you'd use a real sandbox service (LangSmith, OpenShell, Modal, etc.)
    This just demonstrates the pattern.
    """
    
    def __init__(self, container_id: str):
        self._container_id = container_id
    
    @property
    def id(self) -> str:
        return self._container_id
    
    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run a command inside the container — this is the only method you must implement."""
        try:
            result = subprocess.run(
                ["docker", "exec", self._container_id, "bash", "-c", command],
                capture_output=True, text=True,
                timeout=timeout or 60,
            )
            output = result.stdout
            if result.stderr:
                output = f"{output}\n{result.stderr}" if output else result.stderr
            return ExecuteResponse(output=output, exit_code=result.returncode)
        except subprocess.TimeoutExpired:
            return ExecuteResponse(output="Command timed out", exit_code=1)
    
    # upload_files and download_files have default implementations in BaseSandbox
    # that use execute() with base64 encoding. Override only if your sandbox
    # has a more efficient native file transfer mechanism.

print("LocalDockerSandbox defined — demonstrates the BaseSandbox pattern")
print()
print("What BaseSandbox gives you for free (all via execute()):")
print("  ls, read, write, edit, glob, grep")
print()
print("What you implement:")
print("  execute() — required")
print("  id — required (property)")
print("  upload_files() — optional (default uses execute + base64)")
print("  download_files() — optional (default uses execute + base64)")
```

### Using a sandbox backend with `create_deep_agent`

**Pattern 1: Direct sandbox backend**
```python
agent = create_deep_agent(
    model=model,
    backend=LocalDockerSandbox("my-container-id"),
)
```

**Pattern 2: CompositeBackend — sandbox for code, filesystem for memory**
```python
from deepagents.backends import CompositeBackend, FilesystemBackend

def create_backend(runtime):
    return CompositeBackend(
        default=LocalDockerSandbox("my-container-id"),
        routes={
            "/memories/": FilesystemBackend(root_dir="./data", virtual_mode=True),
        },
    )

agent = create_deep_agent(model=model, backend=create_backend)
```

The agent automatically gets an `execute` tool because the backend implements `SandboxBackendProtocol`.

### Key Takeaway
- To build your own sandbox backend, subclass `BaseSandbox` and implement `execute()` — that's it
- `BaseSandbox` derives all file operations (`ls`, `read`, `write`, etc.) from `execute()` via shell commands
- Use `CompositeBackend` to mix sandbox (for code execution) with `FilesystemBackend` (for persistent memory/skills)
- `LangSmithSandbox` is the production-ready implementation; see [openshell-deepagent](https://github.com/langchain-ai/openshell-deepagent) for another example

---
## Summary

| Part | What you learned | Key pattern |
|------|-----------------|-------------|
| **1. Deconstructing** | `create_deep_agent` = `create_agent` + middleware | Middleware stack composition |
| **2. Filesystem** | Tools return `Command(update={...})` to modify state | State reducers + Command pattern |
| **3. Backends** | `WriteResult.files_update` controls state vs. direct persistence | Backend abstraction, `CompositeBackend` |
| **4. Task definitions** | Replace raw string tasks with structured Pydantic input | Custom `StructuredTool` per subagent |
| **5. Workflow subagents** | Any `Runnable` with `messages` can be a subagent | `CompiledSubAgent` contract |
| **6. MCP adapters** | MCP tools become standard `BaseTool` objects | Adapter pattern, stateless RPC |
| **7. Sandboxes** | `SandboxBackendProtocol` gates the `execute` tool | Protocol extension pattern |

### Where to go from here

- **Build your own middleware** — now that you understand the hooks, create middleware for logging, caching, rate limiting, or custom behavior
- **Custom backends** — implement `BackendProtocol` for your storage (S3, database, etc.)
- **Deploy with LangGraph Platform** — use `langgraph dev` to run your agents with a Studio UI
- **Evaluate** — use LangSmith to measure agent quality with datasets and evaluators
