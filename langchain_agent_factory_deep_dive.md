# LangChain Agent Factory Deep Dive

**File:** `/Users/alanxu/projects/langchain/libs/langchain_v1/langchain/agents/factory.py`

**Date:** 2025-12-30

## Overview

The `factory.py` module implements `create_agent()` - a factory function that constructs a **LangGraph state graph** for running LLM agents with:
- **Tool calling loops** (agent calls tools until a stopping condition)
- **Middleware hooks** (before_agent, before_model, after_model, after_agent)
- **Structured output** (forcing LLMs to return specific schemas)
- **Dual sync/async execution**

---

## Core Architecture

### 1. Middleware Composition System (Lines 86-280)

The factory uses a **functional composition pattern** to chain middleware handlers into a single middleware stack.

**Key Functions:**
- `_chain_model_call_handlers()` (lines 86-195) - Composes sync middleware
- `_chain_async_model_call_handlers()` (lines 198-280) - Composes async middleware

**How it works:**
```python
# If you have middleware: [auth, retry, cache]
# They compose as: auth(retry(cache(base_handler)))

def compose_two(outer, inner):
    def composed(request, handler):
        # Create wrapper for inner that normalizes output
        def inner_handler(req):
            return _normalize_to_model_response(inner(req, handler))

        # Outer wraps inner
        return outer(request, inner_handler)
    return composed
```

**Flow:**
1. Request enters `auth` middleware
2. Auth calls its `handler` callback → invokes `retry`
3. Retry calls its `handler` callback → invokes `cache`
4. Cache calls its `handler` callback → invokes base model execution
5. Response bubbles back up through the stack

**Key insight:** Each middleware receives a `handler` callback representing "everything below me". This allows:
- Preprocessing (modify request before calling handler)
- Postprocessing (modify response after calling handler)
- Error handling (try/catch around handler)
- Retry logic (call handler multiple times)

---

### 2. Tool Call Wrapper Composition (Lines 431-538)

Similar pattern for wrapping tool execution:
- `_chain_tool_call_wrappers()` (sync)
- `_chain_async_tool_call_wrappers()` (async)

Allows middleware to intercept tool calls for:
- Logging tool invocations
- Caching tool results
- Adding auth headers
- Rate limiting

---

### 3. Schema Resolution & State Merging (Lines 283-326)

`_resolve_schema()` merges state schemas from multiple middleware into a single TypedDict.

**Example:**
```python
# Middleware A adds: {"user_id": str}
# Middleware B adds: {"session": Session}
# Base state has: {"messages": list[AnyMessage]}

# Result: TypedDict with all fields merged
```

**OmitFromSchema handling:** Fields can be marked to exclude from input/output schemas:
```python
class MyState(TypedDict):
    internal_data: Annotated[str, OmitFromSchema(output=True)]  # Hidden from output
```

---

### 4. Structured Output System (Lines 364-428)

Handles forcing LLMs to return specific schemas (Pydantic models).

**Two strategies:**

**ProviderStrategy** (lines 1044-1052):
- Uses native model capabilities (OpenAI's `response_format` parameter)
- More reliable, but only works with supporting models
- Auto-detected via `_supports_provider_strategy()`

**ToolStrategy** (lines 1054-1077):
- Converts schema to a "fake tool" the model must call
- Works with any tool-calling model
- Agent validates tool call arguments against schema

**Auto-detection flow (lines 1020-1032):**
```python
if isinstance(request.response_format, AutoStrategy):
    if _supports_provider_strategy(model, tools):
        # Use native structured output
        effective_format = ProviderStrategy(schema=...)
    else:
        # Fallback to tool-based approach
        effective_format = ToolStrategy(schema=...)
```

**Error handling (lines 401-428):**
If schema validation fails, middleware can:
- Retry with error message fed back to model
- Raise exception immediately
- Use custom error handling logic

---

## 5. The Main `create_agent()` Function (Lines 541-1482)

This is where everything comes together.

### Phase 1: Initialization (Lines 684-792)

```python
# 1. Convert model string to BaseChatModel
if isinstance(model, str):
    model = init_chat_model(model)

# 2. Setup system message
system_message = SystemMessage(content=system_prompt) if system_prompt else None

# 3. Setup structured output tools
# If response_format is a raw schema, wrap in AutoStrategy
# Convert to ToolStrategy temporarily to calculate tools needed
structured_output_tools: dict[str, OutputToolBinding] = {}

# 4. Setup tool node (executes client-side tools)
tool_node = ToolNode(tools=available_tools, ...)

# 5. Compose middleware wrappers
wrap_model_call_handler = _chain_model_call_handlers([...])
wrap_tool_call_wrapper = _chain_tool_call_wrappers([...])
```

### Phase 2: State Schema Resolution (Lines 852-869)

```python
# Collect all state schemas from middleware
state_schemas = {m.state_schema for m in middleware}
state_schemas.add(base_state)

# Create three schemas:
resolved_state_schema = _resolve_schema(state_schemas, "StateSchema", None)
input_schema = _resolve_schema(state_schemas, "InputSchema", "input")   # Omit input fields
output_schema = _resolve_schema(state_schemas, "OutputSchema", "output") # Omit output fields
```

### Phase 3: Core Nodes (Lines 871-1196)

**Model Node** (lines 1115-1195):
```python
def model_node(state: AgentState, runtime: Runtime) -> dict:
    # 1. Create ModelRequest with current state
    request = ModelRequest(
        model=model,
        tools=default_tools,
        messages=state["messages"],
        ...
    )

    # 2. Execute through middleware stack
    if wrap_model_call_handler:
        response = wrap_model_call_handler(request, _execute_model_sync)
    else:
        response = _execute_model_sync(request)

    # 3. Return state updates
    return {"messages": response.result, "structured_response": ...}
```

**_execute_model_sync** (lines 1089-1113):
```python
def _execute_model_sync(request: ModelRequest) -> ModelResponse:
    # 1. Auto-detect strategy (Provider vs Tool)
    model_, effective_format = _get_bound_model(request)

    # 2. Invoke model
    messages = [system_message, *request.messages] if system_message else request.messages
    output = model_.invoke(messages)

    # 3. Handle structured output
    handled = _handle_model_output(output, effective_format)

    return ModelResponse(
        result=handled["messages"],
        structured_response=handled.get("structured_response")
    )
```

**_handle_model_output** (lines 871-973):
- Extracts structured responses from AI message
- Validates against schema
- Creates ToolMessages for structured output
- Handles retry logic on validation errors

### Phase 4: Graph Construction (Lines 1288-1472)

This is the most complex part. The factory builds a **state machine** with conditional edges.

**Node Types:**
1. `START` - Entry point
2. `{middleware}.before_agent` - Runs once at start
3. `{middleware}.before_model` - Runs before each model call
4. `model` - Calls LLM
5. `{middleware}.after_model` - Runs after each model call
6. `tools` - Executes tool calls in parallel
7. `{middleware}.after_agent` - Runs once at end
8. `END` - Exit point

**Key Destinations:**
- `entry_node` - First node to run (before_agent → before_model → model)
- `loop_entry_node` - Where to loop back (before_model → model, skips before_agent)
- `loop_exit_node` - End of iteration (model or after_model)
- `exit_node` - Final exit (after_agent or END)

**Edge Logic:**

1. **START → entry_node** (line 1316)

2. **model → tools conditional** (lines 1342-1363):
   ```python
   def model_to_tools(state):
       # Priority 1: Explicit jump_to from middleware
       if state.get("jump_to"):
           return _resolve_jump(...)

       # Priority 2: No tool calls → END
       if no tool_calls:
           return END

       # Priority 3: Has pending tool calls → tools (parallel Send)
       if pending_tool_calls:
           return [Send("tools", call) for call in pending_tool_calls]

       # Priority 4: Structured response complete → END
       if "structured_response" in state:
           return END

       # Priority 5: Tool messages injected by middleware → loop back
       return loop_entry_node
   ```

3. **tools → model conditional** (lines 1328-1340):
   ```python
   def tools_to_model(state):
       # Exit if all tools have return_direct=True
       if all tools are return_direct:
           return END

       # Exit if structured output tool executed
       if structured_output_tool in tool_messages:
           return END

       # Continue loop
       return loop_entry_node
   ```

4. **Middleware edges** (lines 1390-1472):
   - Chain middleware in sequence
   - Support `jump_to` for control flow
   - Connect to loop entry/exit points

---

## Data Flow Example

Let's trace a complete execution:

```python
graph = create_agent(
    model="anthropic:claude-3-5-sonnet",
    tools=[weather_tool],
    middleware=[auth_middleware, logging_middleware],
    system_prompt="You are a helpful assistant"
)

result = graph.invoke({"messages": [{"role": "user", "content": "What's the weather in SF?"}]})
```

**Execution flow:**

1. **START** → `auth.before_agent`
2. `auth.before_agent` → `logging.before_agent` (could add API keys to state)
3. `logging.before_agent` → `auth.before_model`
4. `auth.before_model` → `logging.before_model` (could log request)
5. `logging.before_model` → `model`

6. **model node** calls:
   ```python
   request = ModelRequest(messages=[...], tools=[weather_tool])

   # Through middleware stack: auth → logging → base
   response = auth.wrap_model_call(
       request,
       lambda req: logging.wrap_model_call(req, _execute_model_sync)
   )
   ```

7. Model returns: `AIMessage(tool_calls=[{"name": "weather", "args": {"location": "SF"}}])`

8. `model` → conditional edge → `tools` (via Send for parallel execution)

9. **tools node** executes `weather_tool("SF")` → `ToolMessage(content="Sunny, 72°F")`

10. `tools` → conditional edge → `logging.before_model` (loop back)

11. `logging.before_model` → `model`

12. **model node** calls LLM again with tool result

13. Model returns: `AIMessage(content="The weather in SF is sunny and 72°F")` (no tool calls)

14. `model` → conditional edge → `logging.after_model` (no tool calls, so exit loop)

15. `logging.after_model` → `auth.after_model`

16. `auth.after_model` → `logging.after_agent`

17. `logging.after_agent` → `auth.after_agent`

18. `auth.after_agent` → **END**

---

## Design Patterns & Key Insights

### 1. Functional Composition
Middleware is composed functionally (like Redux middleware or Express.js) rather than using inheritance. This allows:
- Order matters (first in list = outermost layer)
- Easy to reason about data flow
- No complex class hierarchies

### 2. Dual Sync/Async
Everything has both sync and async versions:
- `model_node` / `amodel_node`
- `wrap_model_call` / `awrap_model_call`
- Uses `RunnableCallable` to wrap both versions

### 3. Conditional Routing
The graph uses conditional edges to implement the agent loop:
- Check `jump_to` for middleware control
- Check tool calls to continue/exit loop
- Check structured response completion
- Fallback to default behavior

### 4. Parallel Tool Execution
When model calls multiple tools, the graph uses `Send` to execute them in parallel:
```python
[Send("tools", call1), Send("tools", call2), Send("tools", call3)]
```

### 5. Schema Auto-Detection
The `AutoStrategy` pattern allows the factory to choose the best structured output strategy based on model capabilities, falling back gracefully.

### 6. State Machine as a Graph
The entire agent is modeled as a **state graph** where:
- Nodes = actions (call model, run tools, run middleware)
- Edges = transitions (conditional routing logic)
- State = messages list + custom fields

---

## Critical Implementation Details

**Jump Destinations (lines 1288-1314):**
- `entry_node`: First run only (includes before_agent)
- `loop_entry_node`: Loop re-entry (excludes before_agent)
- Prevents before_agent from running on every iteration

**Tool Validation (lines 988-1018):**
- Only validates client-side tools (not provider built-ins)
- Middleware can add tools via `middleware.tools` attribute
- Built-in tools (dict format) can be added dynamically

**Structured Output Tools (lines 725-729):**
- Converted to "fake tools" with specific names
- Added to model binding only with ToolStrategy
- Excluded from normal tool execution routing

**Error Normalization (lines 79-83):**
- Middleware can return `AIMessage` or `ModelResponse`
- All returns normalized to `ModelResponse` for consistency

---

## Summary

This factory is essentially building a **programmable agent runtime** where:
- The execution flow is a directed graph with conditional edges
- Middleware can intercept at any point (before/after agent, before/after model, wrap tool calls)
- The routing logic elegantly handles tool loops, structured output, and control flow
- Both sync and async execution paths are supported
- State is managed through a TypedDict schema that merges middleware requirements

The key innovation is treating the agent as a **state machine** where middleware can:
1. Modify state before/after operations
2. Wrap the model call itself (retry, caching, auth)
3. Wrap individual tool calls
4. Control routing via `jump_to` directives

This provides maximum flexibility while maintaining a clean, functional composition model.
