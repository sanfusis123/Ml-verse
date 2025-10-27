# LangChain Guide

## Table of Contents
1. Core Architecture & Concepts
2. LCEL (LangChain Expression Language)
3. Models & Prompts
4. Retrieval & RAG
5. Agents & Tools
6. Memory & State Management
7. Advanced Topics
8. Practical Scenarios & Code Examples

---

## 1. Core Architecture & Concepts

### Key Packages Understanding
**Question**: Explain the LangChain ecosystem architecture and the purpose of different packages.

**Answer**: LangChain ecosystem consists of:
- **langchain-core**: Contains base abstractions like LLMs, vector stores, retrievers, and the Runnable protocol. No third-party integrations, kept lightweight
- **langchain**: Contains chains, agents, and retrieval strategies that form application's cognitive architecture. Generic implementations not specific to any integration
- **langchain-community**: Third-party integrations maintained by the community with optional dependencies
- **langgraph**: Low-level agent orchestration framework for complex, deterministic workflows with durable execution
- **langserve**: Deploys LangChain Runnables as REST endpoints using FastAPI

**Follow-up**: When would you choose LangChain vs LangGraph?
- Use LangChain for quickly building agents and simple applications
- Use LangGraph for advanced needs requiring deterministic workflows, heavy customization, and controlled latency

---

### The Runnable Protocol
**Question**: What is the Runnable protocol and why is it important?

**Answer**: The Runnable protocol is a standard interface implemented by many LangChain components (chat models, LLMs, output parsers, retrievers, prompt templates). It provides:
- **Consistent invocation**: `invoke()`, `batch()`, `stream()` methods
- **Easy composition**: Chain components using the `|` operator
- **Async support**: `ainvoke()`, `abatch()`, `astream()` methods
- **Standard configuration**: `with_config()`, `with_retry()`, `with_fallbacks()`

**Example**:
```python
chain = prompt | model | output_parser
result = chain.invoke({"input": "Hello"})
```

---

## 2. LCEL (LangChain Expression Language)

### Core Benefits
**Question**: What are the advantages of LCEL over legacy chains like LLMChain?

**Answer**: LCEL provides:
1. **Transparency**: No hidden prompts or implementation details
2. **Optimized parallel execution**: Automatically runs parallel steps (e.g., multiple retrievers)
3. **Retries and fallbacks**: Configurable for any chain part
4. **Streaming support**: Access intermediate results before final output
5. **Async support**: Both sync and async interfaces
6. **Customization**: Greater flexibility as models diversify

**Migration context**: Legacy chains like `ConversationalRetrievalChain` and `RetrievalQA` are deprecated in favor of LCEL implementations.

---

### RunnableParallel & RunnablePassthrough
**Question**: How do RunnableParallel and RunnablePassthrough work?

**Answer**: 
- **RunnableParallel**: Executes multiple runnables in parallel and returns a dict
```python
setup_and_retrieval = RunnableParallel({
    "context": retriever,
    "question": RunnablePassthrough()
})
```
- **RunnablePassthrough**: Passes input unchanged to the next step, useful for preserving original input

**Use case**: In RAG chains, we need both retrieved context and original question for the prompt.

---

### Chain Composition Patterns
**Question**: Demonstrate building a RAG chain using LCEL.

**Answer**:
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

template = """Answer based on context:
{context}

Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)
retriever = vectorstore.as_retriever()
output_parser = StrOutputParser()

setup = RunnableParallel({
    "context": retriever,
    "question": RunnablePassthrough()
})

chain = setup | prompt | model | output_parser
result = chain.invoke("What is LangChain?")
```

**Key Points**:
- Each step transforms data: dict → PromptValue → AIMessage → string
- Can test intermediate steps: `prompt.invoke(input)` or `(prompt | model).invoke(input)`

---

## 3. Models & Prompts

### Prompt Templates
**Question**: What are the different types of prompt templates and when to use each?

**Answer**:

1. **String Prompt Templates** - Simple text formatting:
```python
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template("Tell me about {topic}")
```

2. **Chat Prompt Templates** - For chat models with message lists:
```python
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me about {topic}")
])
```

3. **MessagesPlaceholder** - For dynamic message lists:
```python
from langchain_core.prompts import MessagesPlaceholder
prompt = ChatPromptTemplate([
    ("system", "You are helpful"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
```

**Purpose**: Separate static prompt parts for versioning, serialization, and reuse.

---

### Model Standardization
**Question**: How does LangChain standardize model interfaces?

**Answer**: LangChain provides a uniform interface across different providers:
- Consistent API regardless of provider (OpenAI, Anthropic, etc.)
- Standard input/output formats (messages in, messages out)
- Prevents vendor lock-in - easy provider swapping
- Same methods: `invoke()`, `stream()`, `batch()` across all models

**Example**:
```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Both use identical interface
model1 = ChatOpenAI(model="gpt-4")
model2 = ChatAnthropic(model="claude-3-sonnet")

# Same invocation pattern
response = model1.invoke([("user", "Hello")])
```

---

## 4. Retrieval & RAG

### Retrieval Fundamentals
**Question**: Explain the retrieval architecture in LangChain.

**Answer**: Retrieval systems in LangChain handle different data types:

1. **Unstructured Data**:
   - Vector stores (embedding-based similarity search)
   - Lexical search indexes (keyword-based)

2. **Structured Data**:
   - Relational databases (SQL)
   - Graph databases (knowledge graphs)

3. **Retriever Interface**: All retrievers share common interface:
```python
retriever = vectorstore.as_retriever()
docs = retriever.invoke("query")
```

---

### Vector Store Concepts
**Question**: How do vector stores work and why use them?

**Answer**: 
- **Mechanism**: Use embedding models to compress documents into high-dimensional vectors
- **Search**: Similarity search using cosine similarity or other distance metrics
- **Advantages over keyword search**:
  - Semantic understanding (meaning, not just words)
  - Language-agnostic
  - Handles synonyms and context

**Implementation**:
```python
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings

vectorstore = DocArrayInMemorySearch.from_texts(
    ["doc1 text", "doc2 text"],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
```

---

### Query Analysis
**Question**: What is query analysis and why is it important in RAG?

**Answer**: Query analysis transforms raw user queries into effective search queries:

**Benefits**:
1. **Query Clarification**: Rephrase ambiguous queries
2. **Semantic Understanding**: Capture intent beyond keywords
3. **Query Expansion**: Add related terms/concepts
4. **Complex Query Handling**: Break multi-part questions into sub-queries

**Techniques**:
- Query decomposition using structured output
- Query rewriting with LLMs
- Text-to-SQL for relational databases
- Natural language to graph queries

**Example**:
```python
# Query decomposition
decomposition_prompt = """Break this question into sub-questions:
{question}

Return as list of questions."""
```

---

### Modern RAG Implementation
**Question**: Implement a modern RAG chain with history awareness.

**Answer**:
```python
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Contextualize question with history
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given chat history and question, formulate standalone question"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)

# Answer with context
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using context. {context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Usage
result = rag_chain.invoke({
    "input": "What are they?",
    "chat_history": [("human", "Tell me about agents"), ("ai", "Agents are...")]
})
```

**Key Concepts**:
- History-aware retriever reformulates questions with context
- Prevents irrelevant retrieval from ambiguous references
- Maintains conversational flow

---

### ConversationalRetrievalChain Migration
**Question**: Why was ConversationalRetrievalChain deprecated and how to migrate?

**Answer**: 

**Reasons for deprecation**:
- Hidden internals (two prompts, two LLMs)
- Difficult to customize
- Doesn't leverage LCEL benefits

**Migration pattern**:
```python
# OLD (deprecated)
convo_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever, 
    condense_question_prompt=...,
    combine_docs_chain_kwargs={...}
)

# NEW (LCEL)
history_aware_retriever = create_history_aware_retriever(...)
qa_chain = create_stuff_documents_chain(...)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
```

**Advantages**: Clear internals, easier customization, streaming support

---

## 5. Agents & Tools

### Agent Architecture
**Question**: Explain LangChain's agent architecture and its relationship with LangGraph.

**Answer**: 
- **LangChain Agents**: Pre-built, easy-to-use agent abstraction (< 10 lines of code)
- **Built on LangGraph**: Provides durable execution, streaming, human-in-the-loop, persistence
- **Don't need LangGraph knowledge** for basic agent usage
- **Use LangGraph directly** for advanced customization and complex workflows

**Agent Components**:
1. LLM as reasoning engine
2. Tools for external interactions
3. Prompt engineering for behavior
4. Memory for context retention

---

### Tool Usage Pattern
**Question**: How do agents decide which tools to use?

**Answer**: Agent workflow:
1. **Observation**: Receive user input and current state
2. **Thought**: LLM reasons about what to do
3. **Action**: Select and execute tool
4. **Observation**: Receive tool result
5. **Repeat** until final answer ready

**Example**:
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool

tools = [
    Tool(name="Search", func=search_func, description="Search the web"),
    Tool(name="Calculator", func=calc_func, description="Do math")
]

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
result = agent_executor.invoke({"input": "What is 2+2 and search for cats"})
```

---

## 6. Memory & State Management

### Memory Types
**Question**: What are different memory types in LangChain?

**Answer**:

1. **ConversationBufferMemory**: Stores all messages
```python
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()
```

2. **ConversationBufferWindowMemory**: Keeps last K messages
```python
memory = ConversationBufferWindowMemory(k=5)
```

3. **ConversationSummaryMemory**: Summarizes old messages
```python
memory = ConversationSummaryMemory(llm=llm)
```

4. **ConversationSummaryBufferMemory**: Hybrid approach
```python
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
```

**Usage**: Memory is called at chain start (load variables) and end (save variables)

---

### State Persistence
**Question**: How do you implement persistent state in LangChain applications?

**Answer**: 
- Use LangGraph's checkpointer for durable execution
- Store chat history externally (database, Redis)
- Implement custom memory classes
- Use message history in prompts via MessagesPlaceholder

**Pattern**:
```python
# Store messages
chat_history = []
result = chain.invoke({
    "input": "question",
    "chat_history": chat_history
})
chat_history.extend([
    ("human", "question"),
    ("ai", result["answer"])
])
```

---

## 7. Advanced Topics

### Streaming & Async
**Question**: How do you implement streaming in LCEL chains?

**Answer**:
```python
# Streaming tokens
for chunk in chain.stream({"input": "Hello"}):
    print(chunk, end="", flush=True)

# Async streaming
async for chunk in chain.astream({"input": "Hello"}):
    print(chunk, end="", flush=True)

# Stream events
async for event in chain.astream_events({"input": "Hello"}, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content)
```

**Benefits**: Lower perceived latency, real-time feedback

---

### Callbacks & Tracing
**Question**: Explain the callback system and tracing in LangChain.

**Answer**: 

**Callbacks**: Execute custom code during chain lifecycle
- `on_chain_start`, `on_chain_end`, `on_chain_error`
- `on_llm_start`, `on_llm_end`
- `on_tool_start`, `on_tool_end`

**Tracing**: Recording application steps from input to output
- **Trace**: Series of steps (input → output)
- **Runs**: Individual steps (model call, retrieval, tool use)
- **LangSmith**: Platform for tracing, debugging, evaluation

**Purpose**: Debugging, observability, performance monitoring

---

### Output Parsers
**Question**: When and why use output parsers?

**Answer**: Output parsers transform model output into structured formats:

**Types**:
1. **StrOutputParser**: Extract string from AIMessage
2. **JsonOutputParser**: Parse JSON responses
3. **PydanticOutputParser**: Validate with Pydantic models
4. **StructuredOutputParser**: For structured data

**Note**: Less critical now with tool calling and structured outputs built into modern LLMs

**Example**:
```python
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
chain = prompt | model | parser  # Returns string instead of AIMessage
```

---

### Few-Shot Prompting & Example Selectors
**Question**: How do you implement dynamic few-shot prompting?

**Answer**: Example selectors choose relevant examples based on input:

```python
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

examples = [
    {"input": "What's 2+2?", "output": "4"},
    {"input": "What's the capital of France?", "output": "Paris"}
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    vectorstore_cls,
    k=1
)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Answer questions:",
    suffix="Question: {input}",
    input_variables=["input"]
)
```

**Benefits**: Improves model performance through relevant examples

---

### Retries & Fallbacks
**Question**: How do you implement error handling in LCEL chains?

**Answer**:
```python
# Retries
chain_with_retry = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)

# Fallbacks
chain_with_fallback = primary_chain.with_fallbacks([
    fallback_chain_1,
    fallback_chain_2
])

# Combined
reliable_chain = (
    primary_model
    .with_retry(stop_after_attempt=2)
    .with_fallbacks([backup_model])
)
```

**Benefits**: Reliability at scale, no latency cost with streaming

---

## 8. Practical Scenarios & Code Examples

### Scenario 1: Multi-Query RAG
**Question**: Implement a RAG system that generates multiple query variations for better retrieval.

**Answer**:
```python
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

query_generation_prompt = ChatPromptTemplate.from_template("""
Generate 3 different search queries for: {question}
Return as JSON list.
""")

generate_queries = query_generation_prompt | model | JsonOutputParser()

# Retrieve for each query
from langchain_core.runnables import RunnableParallel

def retrieve_multiple(queries):
    all_docs = []
    for query in queries:
        all_docs.extend(retriever.invoke(query))
    return all_docs

multi_query_retriever = (
    {"queries": generate_queries}
    | RunnableLambda(lambda x: retrieve_multiple(x["queries"]))
)
```

---

### Scenario 2: Conversational Agent with Tools
**Question**: Build an agent that can search and calculate while maintaining conversation.

**Answer**:
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory

tools = [
    Tool(
        name="Search",
        func=search_function,
        description="Search for current information"
    ),
    Tool(
        name="Calculator", 
        func=calculator_function,
        description="Perform calculations"
    )
]

memory = ConversationBufferMemory(memory_key="chat_history")

agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

result = agent_executor.invoke({"input": "Search for GDP of India and calculate 10% of it"})
```

---

### Scenario 3: Conditional Chain Routing
**Question**: Route queries to different chains based on classification.

**Answer**:
```python
from langchain_core.runnables import RunnableBranch

# Classifier
classifier_prompt = ChatPromptTemplate.from_template(
    "Classify this query as 'technical' or 'general': {query}"
)
classifier = classifier_prompt | model | StrOutputParser()

# Different chains
technical_chain = tech_prompt | model | parser
general_chain = general_prompt | model | parser

# Routing
routing_chain = RunnableBranch(
    (lambda x: "technical" in classifier.invoke(x), technical_chain),
    general_chain  # default
)

result = routing_chain.invoke({"query": "How does LCEL work?"})
```

---

### Scenario 4: Document QA with Citation
**Question**: Build a RAG system that cites source documents.

**Answer**:
```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template("""
Answer using context and cite sources.

Context: {context}

Question: {input}

Format: [Answer] (Source: document_id)
""")

# Add metadata to documents
def add_ids(docs):
    for i, doc in enumerate(docs):
        doc.metadata['id'] = f"doc_{i}"
    return docs

retriever_with_ids = retriever | RunnableLambda(add_ids)

document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever_with_ids, document_chain)

result = qa_chain.invoke({"input": "question"})
print(result['answer'])
print(result['context'])  # Source documents with IDs
```

---

## Tips & Best Practices

### What Look For:
1. **Understanding of LCEL**: Can you build chains using the pipe operator?
2. **RAG fundamentals**: Vector stores, embeddings, retrieval strategies
3. **Modern patterns**: Using new APIs instead of deprecated chains
4. **Practical experience**: Real implementation challenges and solutions
5. **Debugging skills**: Using callbacks, tracing, and LangSmith
6. **Performance considerations**: Async, streaming, parallel execution
7. **Production readiness**: Error handling, retries, fallbacks

### Common Pitfalls to Avoid:
- Using deprecated chains (ConversationalRetrievalChain, RetrievalQA, LLMChain)
- Not understanding the difference between LangChain and LangGraph
- Ignoring error handling and retries
- Not leveraging parallel execution in LCEL
- Overlooking the importance of query analysis in RAG
- Forgetting about streaming for better UX

### Key Differentiators for Middle-Level:
- Deep understanding of LCEL composition
- Experience migrating legacy chains
- Knowledge of when to use LangGraph vs LangChain
- Practical RAG optimization techniques
- Understanding of agent architectures and limitations
- Production deployment experience

---

## Additional Resources
- Official LangChain documentation: https://docs.langchain.com
- LangSmith for debugging: https://docs.langchain.com/langsmith
- Migration guides for legacy chains
- LangGraph documentation for advanced orchestration
