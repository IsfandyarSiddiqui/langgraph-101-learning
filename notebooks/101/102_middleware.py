# %% [markdown]
# # LangGraph Advanced Concepts: Middleware & Human-in-the-Loop
# 
# Welcome to LangGraph Advanced Concepts! This notebook builds on the foundations from LangGraph 101 and introduces two powerful patterns for production agents.
# 
# **What you'll learn:**
# - **Human-in-the-Loop** - Pause agents for human review and approval
# - **Middleware** - Modify agent behavior at key points in execution
# - **Tool Review** - Add approval workflows to sensitive tools
# - **Dynamic Behavior** - Adapt agent responses based on context
# 
# **Prerequisites:** Complete `langgraph_101.ipynb` 
# </br>
# </br>
# 
# ---
# </br>
# 
# > **Note:** These patterns are essential for production agents where safety, compliance, and user control are critical.

# %% [markdown]
# ## Setup
# 
# Let's quickly set up our environment.

# %%
# Add project root to Python path so we can import from utils module
import sys
from pathlib import Path
project_root = Path().resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import model from centralized utils module
# This avoids code duplication across notebooks and uses consistent model configuration
from utils.models import model

# Alternative: If you want to define the model inline instead of using centralized config, uncomment below:
# from dotenv import load_dotenv
# load_dotenv(dotenv_path="../.env", override=True)
# from langchain.chat_models import init_chat_model
# model = init_chat_model("openai:o3-mini")

# Note: For other providers (Azure, Bedrock, Vertex AI), update utils/models.py 
# See utils/models.py for detailed instructions on switching LLM providers

# %% [markdown]
# ## Part 1: Human-in-the-Loop with Interrupts
# 
# ### The Problem
# 
# Imagine you're building an agent that can send emails or make purchases. You don't want it to take these actions automatically - you want human approval first!
# 
# **Human-in-the-loop** lets you:
# - Pause execution for review
# - Approve, reject, or edit actions
# - Add safety controls to sensitive operations
# 
# ### How It Works
# 
# 1. Agent encounters an `interrupt()` - execution pauses
# 2. System surfaces information to human
# 3. Human provides input (approve/reject/edit)
# 4. Agent resumes with `Command(resume=...)`

# %% [markdown]
# ### Example 1: Simple Approval Workflow
# 
# Let's start with a simple example - asking for approval before sending an email.

# %%
from langgraph.types import interrupt
from langchain_core.tools import tool

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    
    # Pause for human approval
    approval = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "Do you want to send this email?"
    })
    
    if approval.get("approved"): # Will be true if accepted, false if declined
        # In production, this would actually send the email
        return f" Email sent to {to} with subject '{subject}'"
    else:
        return "Email cancelled by user"

# Test the tool directly
print("Tool created successfully!")
print(f"Tool name: {send_email.name}")
print(f"Tool description: {send_email.description}")

# %% [markdown]
# ### Creating an Agent with Human-in-the-Loop
# 
# Now let's create an agent that uses this tool. **Remember:** Interrupts require a checkpointer!

# %%
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# Create checkpointer for persistence
checkpointer = MemorySaver()

# Create agent with the email tool
agent = create_agent(
    model=model,
    tools=[send_email],
    system_prompt="You are a helpful email assistant. When asked to send emails, use the send_email tool.",
    checkpointer=checkpointer  # Required for interrupts
)

# %% [markdown]
# ### Running Until Interrupt
# 
# Let's run the agent and see it pause for approval:

# %%
from langchain.messages import HumanMessage
from langsmith import uuid7

# Create a unique thread for this conversation
config = {"configurable": {"thread_id": uuid7()}}

# Run the agent and see it pause for approval
result = agent.invoke(
    {
        "messages": [HumanMessage(content="Send an email to alice@example.com with subject 'Meeting Tomorrow' and body 'Let's meet at 3pm.'")]
    },
    config=config
)

# Check if we hit an interrupt

if "__interrupt__" in result:
    print("Agent paused for approval\n")

    interrupt_info = result["__interrupt__"][0]

    print("Interrupt details:")
    print(f"  To: {interrupt_info.value['to']}")
    print(f"  Subject: {interrupt_info.value['subject']}")
    print(f"  Body: {interrupt_info.value['body']}")
    print(f"  Message: {interrupt_info.value['message']}")
else:
    print("Agent completed without interrupt")

# %% [markdown]
# ### Resuming with Approval
# 
# Now let's approve the email and let the agent continue:

# %%
from langgraph.types import Command

# Resume with approval
result = agent.invoke(
    Command(resume={"approved": True}),
    config=config
)

# Print the final response
print("Final response:")
print(result["messages"][-1].content)

# %% [markdown]
# ### Exercise: Try Rejecting the Email
# 
# Run the cells again, but this time reject the email by passing `{"approved": False}`:

# %%
# New thread for rejection example
config_2 = {"configurable": {"thread_id": uuid7()}}

# Run until interrupt
result = agent.invoke(
    {
        "messages": [HumanMessage(content="Send an email to bob@example.com saying 'Hello!'")]
    },
    config=config_2
)

# Resume with rejection
result = agent.invoke(
    Command(resume={"approved": False}),  # Reject the email
    config=config_2
)

print("Final response:")
print(result["messages"][-1].content)

# %% [markdown]
# ## Part 2: Advanced Pattern - Edit Before Execution
# 
# Sometimes you want to **edit** the tool call, not just approve/reject it. Let's enhance our tool:

# %%
@tool
def send_email_v2(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    
    # Pause for human review
    response = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "Review this email. You can approve, reject, or edit it."
    })
    
    # Handle different response types
    if response["type"] == "approve":
        return f"Email sent to {to} with subject '{subject}'"

    elif response["type"] == "reject":
        return "Email cancelled"

    elif response["type"] == "edit":
        # Use edited values
        to = response.get("to", to)
        subject = response.get("subject", subject)
        body = response.get("body", body)
        return f"""Email sent with edits:
                To: {to}
                Subject: {subject}
                Body: {body}"""
    
    return "Unknown response"

# Create new agent with enhanced tool
agent_v2 = create_agent(
    model=model,
    tools=[send_email_v2],
    system_prompt="You are a helpful email assistant.",
    checkpointer=MemorySaver()
)

# %%
# Run and edit the email
config_3 = {"configurable": {"thread_id": uuid7()}}

# Run until interrupt
result = agent_v2.invoke(
    {
        "messages": [HumanMessage(content="Send an email to team@example.com about the meeting")]
    },
    config=config_3
)

print("Paused for review...\n")

# %% [markdown]
# Now lets edit the email subject to make it URGENT meeting!

# %%
# Resume with edits
result = agent_v2.invoke(
    Command(resume={
        "type": "edit",
        "subject": "URGENT: Meeting Today at 2pm",  # We have edited the email subject
        "body": "This is the edited email body with more details."
    }),
    config=config_3
)

print("Final response:")
print(result["messages"][-1].content)

# %% [markdown]
# ## Part 3: Introduction to Middleware
# 
# **Middleware** provides fine-grained control over the agent loop. It lets you:
# - Inspect state before/after model calls
# - Modify model requests dynamically
# - Add custom logic at key execution points
# 
# ### The Agent Loop
# 
# ```
# Input --> [before_model] --> [wrap_model_call] --> Model --> [after_model] --> Tools --> ...
# ```
# 
# Middleware hooks into this loop:
# - **`before_model`** - Runs before model execution, can update state
# - **`wrap_model_call`** - Wraps the model call, control when/how the model is invoked
# - **`after_model`** - Runs after model execution, before tools
# 
# ### Two Hook Styles
# 
# **Node-style hooks** run sequentially:
# - `before_agent`, `before_model`, `after_model`, `after_agent`
# - Good for logging, validation, state updates
# 
# **Wrap-style hooks** intercept execution:
# - `wrap_model_call`, `wrap_tool_call`
# - Full control over handler calls
# - Good for retries, caching, transformation

# %% [markdown]
# ### Example 1: Dynamic System Prompt
# 
# Let's create middleware that changes the system prompt based on the user's role:

# %%
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from typing import TypedDict

# Define context schema
class Context(TypedDict):
    user_role: str

# Create middleware using decorator
@dynamic_prompt
def dynamic_prompt_middleware(request: ModelRequest) -> str:
    """Adjust system prompt based on user role."""
    
    user_role = request.runtime.context.get("user_role", "general")
    
    if user_role == "expert":
        return "You are an AI assistant for experts. Provide detailed technical responses with code examples."
    elif user_role == "beginner":
        return "You are an AI assistant for beginners. Explain concepts simply, avoid jargon."
    else:
        return "You are a helpful AI assistant."

# %%
from langchain_core.tools import tool

@tool
def explain_concept(concept: str) -> str:
    """Explain a programming concept."""
    explanations = {
        "async": "Asynchronous programming allows code to run without blocking.",
        "recursion": "Recursion is when a function calls itself."
    }
    return explanations.get(concept.lower(), "Concept not found.")

# Create agent with middleware
agent_with_middleware = create_agent(
    model=model,
    tools=[explain_concept],
    middleware=[dynamic_prompt_middleware],
    context_schema=Context
)

# %% [markdown]
# ### Testing Different User Roles
# 
# Let's see how the agent responds differently based on user role:

# %%
# Expert user
print("=" * 50)
print("EXPERT USER")
print("=" * 50)

result = agent_with_middleware.invoke(
    {"messages": [HumanMessage(content="Explain async programming")]},
    context={"user_role": "expert"}
)
print(result["messages"][-1].content)
print()

# Beginner user
print("=" * 50)
print("BEGINNER USER")
print("=" * 50)

result = agent_with_middleware.invoke(
    {"messages": [HumanMessage(content="Explain async programming")]},
    context={"user_role": "beginner"}
)
print(result["messages"][-1].content)

# %% [markdown]
# ### Example 2: Custom Middleware - Request Logger
# 
# Middleware lets you hook into the agent loop and see what's happening at each step. This is incredibly useful for debugging and understanding how your agent works.
# 
# **The Agent Loop:**
# User Input --> [before_model] --> [wrap_model_call] --> Model --> [after_model] --> Tools --> ...
# 
# **What we'll build:**
# A logger that prints information at each step:
# - **Before model** - How many messages are in the conversation?
# - **Wrap model call** - Which model and tools are being used?
# - **After model** - Did the model call a tool or give a final answer?
# 
# This is like adding debug `print()` statements, but in a clean, reusable way!
# 
# Let's create middleware that logs model requests for debugging:

# %%
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from typing import Any, Callable

class RequestLoggerMiddleware(AgentMiddleware):
    """Logs all model requests for debugging."""
    
    def before_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
        """Log before model execution."""
        message_count = len(state.get("messages", []))
        print(f"[BEFORE MODEL] Processing {message_count} messages")
        return None  # Don't modify state
    
    def wrap_model_call(
        self, 
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Log the model request details and call the handler."""
        print(f"  [MODEL REQUEST]")
        print(f"   Model: {request.model if hasattr(request, 'model') else 'default'}")
        print(f"   Tools available: {len(request.tools) if request.tools else 0}")
        
        # Call the actual model handler
        return handler(request)
    
    def after_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
        """Log after model execution."""
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print(f" [AFTER MODEL] Model requested {len(last_message.tool_calls)} tool call(s)")
        else:
            print(f" [AFTER MODEL] Model provided final response")
        return None  # Don't modify state

# %%
# Create agent with logger middleware
agent_with_logger = create_agent(
    model=model,
    tools=[explain_concept],
    middleware=[RequestLoggerMiddleware()],
)

# %% [markdown]
# ### What to Expect
# 
# When we run the agent with the logger, you'll see the execution flow in real-time:
# 
# **First iteration:**
# 1. `[BEFORE MODEL]` - Shows how many messages we're starting with
# 2. `[MODEL REQUEST]` - Shows which model and tools are available (from wrap_model_call)
# 3. `[AFTER MODEL]` - The model decides to call the `explain_concept` tool
# 
# **Second iteration (after tool execution):**
# 1. `[BEFORE MODEL]` - Now we have more messages (including tool result)
# 2. `[MODEL REQUEST]` - Model info again
# 3. `[AFTER MODEL]` - Model provides the final answer (no more tools needed)
# 
# This gives you a detailed view into your agent's decision-making process.
# 
# Let's run it:

# %%
# Run and observe the logs
print("\n" + "=" * 50)
print("RUNNING AGENT WITH LOGGER")
print("=" * 50 + "\n")

result = agent_with_logger.invoke({
    "messages": [{"role": "user", "content": "Explain recursion"}]
})

print("\n" + "=" * 50)
print("FINAL RESPONSE")
print("=" * 50)
print(result["messages"][-1].content)

# %% [markdown]
# ## Part 4: Combining Middleware and Human-in-the-loop
# 
# Let's combine human-in-the-loop AND middleware for a production-ready agent:

# %%
# Sensitive tool that needs approval
@tool
def delete_database(database_name: str) -> str:
    """Delete a database. THIS IS DANGEROUS!"""
    
    response = interrupt({
        "action": "delete_database",
        "database_name": database_name,
        "warning": "This will permanently delete the database!",
        "message": "Are you absolutely sure?"
    })
    
    if response.get("confirmed"):
        return f"Database '{database_name}' has been deleted (simulation)"
    else:
        return "Database deletion cancelled"

# Middleware to track dangerous operations
class SafetyMiddleware(AgentMiddleware):
    """Add safety checks and logging."""
    
    name = "safety_checker"
    
    def after_model(self, state: AgentState) -> dict[str, Any] | None:
        """Check for dangerous tool calls."""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if "delete" in tool_call["name"].lower():
                    print("   [SAFETY] Dangerous operation detected!")
                    print(f"   Tool: {tool_call['name']}")
                    print(f"   Args: {tool_call['args']}")
        
        return None

# Create production agent
production_agent = create_agent(
    model=model,
    tools=[delete_database],
    middleware=[SafetyMiddleware()],
    checkpointer=MemorySaver()
)

# %% [markdown]
#   ### What to Expect: Layered Safety in Action
# 
#   When we attempt a dangerous operation, you'll see **both** safety mechanisms activate:
# 
#   **Layer 1 - Middleware Detection:**
#   - `[SAFETY] Dangerous operation detected!` - Middleware spots the delete operation
#   - Logs the tool name and arguments for audit trails
# 
#   **Layer 2 - Human Approval (Interrupt):**
#   - Agent execution pauses at the `interrupt()`
#   - Warning message displayed to human reviewer
#   - Execution won't continue until explicit approval
# 
#   **This is defense-in-depth:** Middleware monitors ALL operations, while interrupts enforce human approval for critical actions.

# %%
# Test the combined pattern
config_4 = {"configurable": {"thread_id": uuid7()}}

print("\n" + "=" * 50)
print("DANGEROUS OPERATION ATTEMPT")
print("=" * 50 + "\n")

# Run until interrupt
result = production_agent.invoke(
    {
        "messages": [HumanMessage(content="Delete the production_db database")]
    },
    config=config_4
)

if "__interrupt__" in result:
    interrupt_info = result["__interrupt__"][0]
    print("\n  Human approval required:")
    print(f"   {interrupt_info.value['warning']}")
    print(f"   Database: {interrupt_info.value['database_name']}")

print("\n(In a real app, a human would review this before proceeding)")

# %% [markdown]
# ## Key Takeaways
# 
# ### Human-in-the-Loop (Interrupts)
# - Use `interrupt()` to pause execution
# - Requires a `checkpointer` for persistence
# - Resume with `Command(resume=value)`
# - Perfect for approval workflows and sensitive operations
# 
# ### Middleware
# - **Node-style hooks**: `before_model`, `after_model` - Sequential logic, validation, logging
# - **Wrap-style hooks**: `wrap_model_call`, `wrap_tool_call` - Full control, retries, transformation
# - **Decorators**: `@dynamic_prompt`, `@before_model`, `@wrap_model_call` for quick middleware
# - **Classes**: Subclass `AgentMiddleware` for complex, reusable components
# 
# ### When to Use What?
# 
# **Use Interrupts when:**
# - You need human approval for actions
# - You want to review/edit tool calls
# - You need to validate user input
# 
# **Use Middleware when:**
# - You need to modify agent behavior dynamically
# - You want to add logging/monitoring
# - You need to enforce policies (token limits, safety checks)
# - You want to personalize responses based on context
# 
# **Node-style vs Wrap-style:**
# - Node-style for sequential operations (logging, validation)
# - Wrap-style for control flow (retry, fallback, caching)

# %% [markdown]
# ## Practice Exercise (Optional)
# 
# Try building an agent that:
# 1. Has a tool to make a purchase
# 2. Uses middleware to check if the purchase amount is over $1000
# 3. If over $1000, uses an interrupt to require approval
# 4. If under $1000, processes automatically
# 
# Hint: Combine `before_model` middleware with conditional `interrupt()` logic!

# %%
# Your code here!
# Challenge: Build the purchase approval agent

# @tool
# def make_purchase(item: str, amount: float) -> str:
#     ...

# class PurchaseLimitMiddleware(AgentMiddleware):
#     ...

# %% [markdown]
# ## Next Steps
# 
# You now have powerful tools for building production agents!
# 
# **Continue your journey:**
# 1.  Check out `multi_agent.ipynb` for multi-agent systems
# 2.  Explore built-in middleware (Summarization, Anthropic Prompt Caching)
# 3.  Build your own custom middleware for your use case
# 4.  Add LangSmith for debugging and monitoring
# 
# **Resources:**
# - [Middleware Documentation](https://docs.langchain.com/oss/python/langchain/middleware)
# - [Human-in-the-Loop Guide](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
# - [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
# 
# </br>
# </br>
# 


