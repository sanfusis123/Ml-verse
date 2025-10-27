# LangGraph

## Table of Contents
1. Core Concepts
2. State Management
3. Graph Architecture
4. Persistence & Memory
5. Human-in-the-Loop
6. Advanced Topics
7. Production Considerations
8. Questions & Answers

---

## 1. Core Concepts

### What is LangGraph?
LangGraph is a low-level framework for building stateful, multi-step agent workflows as cyclical graphs. Unlike high-level frameworks like CrewAI, LangGraph provides fine-grained control over agent orchestration without abstracting away prompts or architecture.

**Key Philosophy**: Message-passing system inspired by Google's Pregel algorithm

### Core Benefits
- **Durable Execution**: Agents persist through failures and resume from checkpoints
- **Human-in-the-Loop**: Inspect and modify agent state at any point
- **Comprehensive Memory**: Short-term (thread-level) and long-term (cross-thread) memory
- **Debugging**: Deep visibility with LangSmith integration
- **Production-Ready**: Scalable infrastructure for stateful workflows

### Three Fundamental Components

#### 1. **State**
- Shared data structure representing the current snapshot
- Can be TypedDict or Pydantic model
- Updated by nodes via reducer functions
- Passed along edges between nodes

```python
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]  # Append to list
    user_input: str  # Override value
    counter: int
```

#### 2. **Nodes**
- Functions that perform actual work
- Receive current state as input
- Return updated state (partial updates)
- Can contain LLM calls, tools, or pure Python logic
- Completely modular

```python
def my_node(state: State) -> dict:
    # Process state
    return {"counter": state["counter"] + 1}
```

#### 3. **Edges**
- Determine execution flow between nodes
- **Normal Edges**: Fixed transitions (A → B)
- **Conditional Edges**: Dynamic routing based on logic
- **START/END**: Special nodes marking graph boundaries

---

## 2. State Management

### State Schema Definition

**Two Update Methods**:
1. **Override**: Replace existing value completely
2. **Reduce**: Aggregate/append to existing value

```python
from typing_extensions import Annotated
from langgraph.graph.message import add_messages

class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]  # Reducer function
    user_id: str  # Override
```

### Reducer Functions
Custom functions to control state aggregation:

```python
def custom_reducer(existing: list, new: int | None) -> list:
    if new is not None:
        return existing + [new]
    return existing

class State(TypedDict):
    values: Annotated[list, custom_reducer]
```

### State Best Practices
- Keep state minimal and focused
- Use proper type hints for validation
- Choose appropriate reducer for each field
- Consider state size for performance

---

## 3. Graph Architecture

### Building a StateGraph

```python
from langgraph.graph import StateGraph, START, END

# 1. Define State
class MyState(TypedDict):
    input: str
    output: str

# 2. Create Graph
builder = StateGraph(MyState)

# 3. Add Nodes
builder.add_node("process", process_node)
builder.add_node("validate", validate_node)

# 4. Add Edges
builder.add_edge(START, "process")
builder.add_edge("process", "validate")
builder.add_edge("validate", END)

# 5. Compile
graph = builder.compile()
```

### Edge Types

#### Normal Edge
```python
# Always go from node_a to node_b
builder.add_edge("node_a", "node_b")
```

#### Conditional Edge
```python
def route_decision(state: State) -> str:
    if state["score"] > 0.8:
        return "high_confidence"
    return "low_confidence"

builder.add_conditional_edge(
    "classifier",
    route_decision,
    {
        "high_confidence": "respond",
        "low_confidence": "human_review",
    }
)
```

#### Sequence Edge
```python
# Execute nodes in order
builder.add_sequence(["node1", "node2", "node3"])
```

### Super-Steps
- One iteration over graph nodes
- Nodes running in parallel = same super-step
- Nodes running sequentially = different super-steps
- Nodes start inactive, become active when receiving messages

---

## 4. Persistence & Memory

### Short-Term Memory (Thread-Level)

**Checkpointer**: Saves state at each super-step

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# Development
checkpointer = InMemorySaver()

# Production
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

graph = builder.compile(checkpointer=checkpointer)
```

### Thread Management

```python
# Start conversation
config = {"configurable": {"thread_id": "user_123"}}
result = graph.invoke({"messages": [user_message]}, config)

# Continue conversation (same thread)
result = graph.invoke({"messages": [next_message]}, config)

# New conversation (different thread)
new_config = {"configurable": {"thread_id": "user_456"}}
```

### Long-Term Memory (Cross-Thread)

**Store Interface**: Share information across threads

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# Save memory across threads
namespace = ("user_123", "preferences")
store.put(namespace, "theme", {"value": "dark_mode"})

# Retrieve in different thread
data = store.get(namespace, "theme")

graph = builder.compile(checkpointer=checkpointer, store=store)
```

### Checkpointer Types
- **InMemorySaver**: Experimentation only
- **SqliteSaver**: Local workflows, development
- **PostgresSaver**: Production deployments
- **RedisSaver**: High-performance, distributed systems
- **AsyncSqliteSaver**: Async operations

### Key Capabilities
1. **Session Memory**: Resume conversations
2. **Error Recovery**: Continue from last successful checkpoint
3. **Human-in-the-Loop**: Interrupt and modify state
4. **Time Travel**: Replay from any checkpoint

---

## 5. Human-in-the-Loop

### Interrupts

**Static Interrupts**: Define at compile time
```python
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review"],  # Pause before this node
    interrupt_after=["critical_action"]  # Pause after this node
)
```

**Dynamic Interrupts**: Interrupt from within nodes
```python
from langgraph.types import interrupt

def approval_node(state: State):
    # Request human input
    user_input = interrupt("Please approve this action")
    return {"approval": user_input}
```

### Resume Execution

```python
from langgraph.types import Command

# After interrupt, resume with user input
config = {"configurable": {"thread_id": "123"}}
result = graph.invoke(
    Command(resume="User approved"),
    config
)
```

### Use Cases
- Approval workflows
- Data validation
- Sensitive operations
- Complex decision points
- User feedback collection

---

## 6. Advanced Topics

### Subgraphs
Encapsulate graph sections for reusability

```python
# Create subgraph
sub_builder = StateGraph(SubState)
sub_builder.add_node("sub_node", sub_function)
subgraph = sub_builder.compile()

# Use as node in parent graph
parent_builder.add_node("sub_workflow", subgraph)
```

**Benefits**:
- Code reusability
- Team collaboration
- Logical separation
- Independent testing

### Streaming

```python
# Stream state updates
for chunk in graph.stream(inputs, config, stream_mode="updates"):
    print(chunk)

# Stream values (full state)
for chunk in graph.stream(inputs, config, stream_mode="values"):
    print(chunk)

# Stream events (including LLM tokens)
async for event in graph.astream_events(inputs, config):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"])
```

**Stream Modes**:
- `updates`: Only changed state keys
- `values`: Full state after each step
- `messages`: Message-by-message
- `events`: Internal events (LLM tokens, tool calls)

### Command Object
Control execution flow dynamically

```python
from langgraph.types import Command

def router_node(state: State):
    if condition:
        return Command(goto="node_a", update={"key": "value"})
    return Command(goto="node_b")
```

### Breakpoints
Step-by-step debugging

```python
graph = builder.compile(
    checkpointer=checkpointer,
    debug=True
)

# Execute step by step
config = {"configurable": {"thread_id": "debug_session"}}
snapshot = graph.get_state(config)
print(f"Next: {snapshot.next}")
```

### Time Travel
Access historical states

```python
# Get checkpoint history
config = {"configurable": {"thread_id": "123"}}
history = list(graph.get_state_history(config))

# Replay from specific checkpoint
checkpoint_config = {
    "configurable": {
        "thread_id": "123",
        "checkpoint_id": history[5].config["checkpoint_id"]
    }
}
result = graph.invoke(inputs, checkpoint_config)
```

---

## 7. Production Considerations

### Error Handling

```python
def robust_node(state: State):
    try:
        result = risky_operation()
        return {"result": result, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}
```

### Rate Limiting & Retries

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def api_call_node(state: State):
    # API call with retries
    pass
```

### Monitoring with LangSmith

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"

# All graph executions now traced
graph.invoke(inputs, config)
```

### Scaling Considerations
- Use async checkpointers for high throughput
- Implement connection pooling for databases
- Consider Redis for distributed systems
- Monitor state size (keep under 5MB per checkpoint)
- Batch operations when possible

### Security
- Encrypt sensitive state data
- Use environment variables for credentials
- Validate inputs at entry points
- Implement proper authentication
- Audit checkpoint access

---

## 8. Questions & Answers

### Basic Questions

**Q1: What is the difference between LangGraph and LangChain Agents?**

**A:** LangChain Agents provide high-level, pre-built architectures for common agent patterns with abstractions that hide implementation details. LangGraph is a low-level framework that exposes the underlying infrastructure for agent orchestration. It provides:
- Explicit control over state management
- Custom node and edge definitions
- Durable execution with checkpointing
- Advanced features like human-in-the-loop and time travel

LangChain Agents are built *on top* of LangGraph for common use cases.

---

**Q2: Explain the concept of "super-steps" in LangGraph.**

**A:** Super-steps are discrete iterations over graph nodes, inspired by Google's Pregel system:
- Nodes start inactive and become active upon receiving messages
- All nodes running in parallel belong to the same super-step
- Sequential nodes belong to different super-steps
- At the end of each super-step, nodes with no incoming messages vote to halt
- Checkpointers save state at each super-step boundary

---

**Q3: What are reducer functions and when would you use them?**

**A:** Reducer functions control how state updates are aggregated:
- Signature: `(existing_value, new_value) -> updated_value`
- Used when multiple nodes update the same state key
- Common patterns:
  - `operator.add` for lists (append)
  - `add_messages` for conversation history
  - Custom reducers for complex aggregation
- Without a reducer, updates override previous values

---

### Intermediate Questions

**Q4: How would you implement a conditional routing in LangGraph?**

**A:**
```python
def route_function(state: State) -> str:
    """Determine next node based on state"""
    if state.get("needs_human_review"):
        return "human_review"
    elif state.get("confidence") < 0.7:
        return "retry"
    else:
        return "complete"

builder.add_conditional_edge(
    "classifier",  # Source node
    route_function,  # Routing logic
    {
        "human_review": "human_node",
        "retry": "retry_node",
        "complete": END
    }
)
```

---

**Q5: What's the difference between short-term and long-term memory in LangGraph?**

**A:**

**Short-Term Memory (Thread-level)**:
- Managed by checkpointers
- Scoped to a single thread/conversation
- Automatic state persistence per super-step
- Accessed through thread_id in config

**Long-Term Memory (Cross-thread)**:
- Managed by Store interface
- Shared across multiple threads
- Explicit save/retrieve operations
- Namespaced by custom keys (e.g., user_id)
- Useful for user preferences, knowledge bases

---

**Q6: Explain the graph compilation process and why it's necessary.**

**A:** Compilation (`builder.compile()`) performs:

1. **Validation**:
   - Check for orphaned nodes
   - Verify edge connectivity
   - Ensure START/END nodes exist

2. **Configuration**:
   - Attach checkpointer
   - Set interrupt points
   - Configure store

3. **Optimization**:
   - Create execution plan
   - Set up message passing
   - Initialize state management

4. **Output**:
   - Returns CompiledGraph (Runnable)
   - Exposes `.invoke()`, `.stream()`, `.astream()` methods

---

### Advanced Questions

**Q7: Design a multi-agent system with LangGraph where agents collaborate on a research task.**

**A:**
```python
class ResearchState(TypedDict):
    query: str
    search_results: Annotated[list, operator.add]
    summaries: Annotated[list, operator.add]
    final_report: str
    current_agent: str

def router(state: ResearchState) -> str:
    if not state.get("search_results"):
        return "searcher"
    elif len(state.get("summaries", [])) < 3:
        return "summarizer"
    else:
        return "writer"

builder = StateGraph(ResearchState)
builder.add_node("searcher", search_agent)
builder.add_node("summarizer", summary_agent)
builder.add_node("writer", writing_agent)

builder.add_edge(START, "router")
builder.add_conditional_edge("router", router, {
    "searcher": "searcher",
    "summarizer": "summarizer",
    "writer": "writer"
})
builder.add_edge("searcher", "router")
builder.add_edge("summarizer", "router")
builder.add_edge("writer", END)
```

---

**Q8: How would you implement graceful error recovery in a production LangGraph application?**

**A:**
```python
class ResilientState(TypedDict):
    data: dict
    error_count: int
    last_error: str
    retry_strategy: str

def error_aware_node(state: ResilientState):
    try:
        result = process_data(state["data"])
        return {
            "data": result,
            "error_count": 0,
            "last_error": None
        }
    except Exception as e:
        error_count = state.get("error_count", 0) + 1
        
        if error_count >= 3:
            # Route to error handling
            return {
                "error_count": error_count,
                "last_error": str(e),
                "retry_strategy": "manual_intervention"
            }
        else:
            # Automatic retry
            return {
                "error_count": error_count,
                "last_error": str(e),
                "retry_strategy": "automatic_retry"
            }

# Use with checkpointer for durability
graph = builder.compile(
    checkpointer=PostgresSaver.from_conn_string(DB_URI)
)
```

**Key strategies**:
- Checkpointing preserves state before failures
- Error counts prevent infinite loops
- Conditional routing to error handlers
- Dead letter queues for manual review
- Alerting and monitoring integration

---

**Q9: Compare different checkpointer implementations and their trade-offs.**

**A:**

| Checkpointer | Use Case | Pros | Cons |
|--------------|----------|------|------|
| **InMemorySaver** | Development, testing | Fast, simple setup | Lost on restart, not scalable |
| **SqliteSaver** | Local workflows, demos | Persistent, file-based | Single-machine, limited concurrency |
| **PostgresSaver** | Production, distributed | Scalable, ACID transactions | Requires DB management |
| **RedisSaver** | High-throughput, real-time | Very fast, distributed | Memory-based, cost at scale |
| **AsyncSqliteSaver** | Async applications | Non-blocking I/O | Still single-machine |

**Selection criteria**:
- **Development**: InMemorySaver
- **Prototypes**: SqliteSaver
- **Production**: PostgresSaver (default) or RedisSaver (high-performance)
- **Async**: Async variants
- **Multi-region**: PostgresSaver with replication

---

**Q10: How would you implement dynamic graph modification based on runtime conditions?**

**A:**
```python
from langgraph.types import Command

def adaptive_router(state: State):
    # Analyze state complexity
    if state["complexity_score"] > 0.8:
        # Add additional validation node dynamically
        return Command(
            goto=["validation_1", "validation_2"],  # Parallel
            update={"validation_mode": "strict"}
        )
    else:
        return Command(
            goto="fast_path",
            update={"validation_mode": "standard"}
        )

# Use Command for dynamic control flow
def validator(state: State):
    result = validate(state["data"])
    if not result["passed"]:
        # Dynamically loop back
        return Command(
            goto="preprocessing",
            update={"retry_count": state.get("retry_count", 0) + 1}
        )
    return {"validation_result": result}
```

**Techniques**:
- Command object for dynamic routing
- State-based decision making
- Parallel node execution
- Conditional loops
- Runtime graph introspection

---

### Scenario-Based Questions

**Q11: You need to build a customer support agent that remembers user preferences across sessions. How would you architect this?**

**A:**
```python
from langgraph.store.memory import InMemoryStore

class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    context: dict

def load_user_context(state: SupportState, store):
    """Load long-term memory"""
    namespace = (state["user_id"], "preferences")
    prefs = store.get(namespace, "user_preferences")
    history = store.get(namespace, "past_issues")
    
    return {
        "context": {
            "preferences": prefs,
            "history": history
        }
    }

def agent_node(state: SupportState, store):
    """Agent with context awareness"""
    system_prompt = f"""
    User preferences: {state['context']['preferences']}
    Past issues: {state['context']['history']}
    
    Use this context to personalize responses.
    """
    # LLM call with context
    
def save_interaction(state: SupportState, store):
    """Update long-term memory"""
    namespace = (state["user_id"], "preferences")
    store.put(namespace, "last_interaction", {
        "timestamp": datetime.now(),
        "issue": state["messages"][-2].content,
        "resolution": state["messages"][-1].content
    })

builder = StateGraph(SupportState)
builder.add_node("load_context", load_user_context)
builder.add_node("agent", agent_node)
builder.add_node("save", save_interaction)

graph = builder.compile(
    checkpointer=PostgresSaver.from_conn_string(DB_URI),
    store=InMemoryStore()
)
```

---

**Q12: Implement a human-in-the-loop approval workflow for sensitive operations.**

**A:**
```python
def sensitive_operation_node(state: State):
    # Prepare operation details
    operation_details = {
        "action": "delete_records",
        "count": state["record_count"],
        "impact": "high"
    }
    
    # Dynamic interrupt for approval
    approval = interrupt({
        "type": "approval_required",
        "details": operation_details,
        "options": ["approve", "reject", "modify"]
    })
    
    if approval == "approve":
        # Execute operation
        result = execute_deletion()
        return {"status": "completed", "result": result}
    elif approval == "reject":
        return {"status": "cancelled"}
    else:
        # Return to previous step for modification
        return Command(
            goto="modify_parameters",
            update={"status": "modification_requested"}
        )

graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_after=["sensitive_operation"]  # Pause after execution
)

# Usage
config = {"configurable": {"thread_id": "approval_123"}}
result = graph.invoke(initial_input, config)

# Human reviews and responds
resumed = graph.invoke(
    Command(resume="approve"),
    config
)
```

---

## Key Takeaway

### Core Competencies to Demonstrate:
1. **Understanding of state management** and reducer functions
2. **Graph architecture** design with proper edge types
3. **Persistence strategies** (checkpointers vs stores)
4. **Human-in-the-loop** patterns and use cases
5. **Production considerations** (error handling, scaling, monitoring)

### Common Patterns to Know:
- Router pattern (conditional routing)
- Human-in-the-loop workflows
- Multi-agent collaboration
- Error recovery and retries
- Memory management (short + long term)

### Red Flags to Avoid:
- Not understanding state updates (override vs reduce)
- Forgetting to compile graph before use
- Ignoring error handling in nodes
- Not considering state size for performance
- Misunderstanding thread_id scope

### Best Practices:
- Keep nodes modular and testable
- Use type hints for state validation
- Implement proper error boundaries
- Monitor with LangSmith in production
- Test with different checkpointers
- Document graph architecture
- Use subgraphs for complex workflows

---

## Additional Resources

- **Official Docs**: https://docs.langchain.com/oss/python/langgraph/overview
- **GitHub**: https://github.com/langchain-ai/langgraph
- **LangSmith**: https://www.langchain.com/langsmith
- **Community**: LangChain Discord server
- **Tutorials**: LangChain YouTube channel

---
