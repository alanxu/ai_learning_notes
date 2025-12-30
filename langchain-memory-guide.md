# LangChain Memory Management: Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [Short-Term Memory Types](#short-term-memory-types)
3. [Long-Term Memory Solutions](#long-term-memory-solutions)
4. [Memory Patterns & Architectures](#memory-patterns--architectures)
5. [Implementation Examples](#implementation-examples)
6. [Comparison Matrix](#comparison-matrix)
7. [Best Practices](#best-practices)
8. [Advanced Techniques](#advanced-techniques)

---

## Overview

LangChain provides a flexible memory system to maintain context across conversations and interactions. Memory components allow LLMs to:
- Remember past interactions
- Maintain conversation context
- Store and retrieve relevant information
- Build knowledge over time

### Memory Lifecycle
```
Input → Memory Read → LLM Processing → Memory Write → Output
         ↓                                ↓
    [Retrieve Context]           [Store New Info]
```

---

## Short-Term Memory Types

### 1. ConversationBufferMemory

**Description**: Stores the complete conversation history in a simple buffer.

**Characteristics**:
- Unlimited history retention
- No compression or filtering
- Simple implementation
- Memory grows unbounded

**Use Cases**:
- Short conversations
- Development/testing
- Complete context required

**Pros**:
- Full conversation context
- No information loss
- Simple to understand

**Cons**:
- Token limit issues with long conversations
- No scaling mechanism
- Inefficient for large histories

**Implementation**:
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context({"input": "Hi, I'm Alice"}, {"output": "Hello Alice!"})
memory.load_memory_variables({})
```

---

### 2. ConversationBufferWindowMemory

**Description**: Maintains a sliding window of the K most recent interactions.

**Characteristics**:
- Fixed-size window (e.g., last 5 exchanges)
- Automatically discards oldest messages
- Predictable memory usage
- Loses older context

**Use Cases**:
- Chatbots with recent context focus
- Customer support (current issue)
- Task-oriented conversations

**Pros**:
- Bounded memory usage
- Always includes most recent context
- Simple configuration

**Cons**:
- Loses historical information
- Hard cutoff (no gradual decay)
- May miss important early context

**Implementation**:
```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5)  # Keep last 5 interactions
```

**Configuration**:
- `k`: Number of recent interactions to retain

---

### 3. ConversationTokenBufferMemory

**Description**: Keeps conversation history within a specified token limit.

**Characteristics**:
- Token-aware pruning
- Dynamic message removal
- Works with token budgets
- Uses tokenizer to count

**Use Cases**:
- API rate-limited scenarios
- Cost optimization
- Model context window management

**Pros**:
- Precise token control
- Adapts to message length
- Prevents token overflow

**Cons**:
- Requires tokenizer configuration
- Still loses old information
- May cut mid-conversation

**Implementation**:
```python
from langchain.memory import ConversationTokenBufferMemory
from langchain.llms import OpenAI

llm = OpenAI()
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=2000)
```

**Configuration**:
- `max_token_limit`: Maximum tokens to retain
- `llm`: LLM instance for tokenization

---

### 4. ConversationSummaryMemory

**Description**: Progressively summarizes conversation history using an LLM.

**Characteristics**:
- Creates summaries of past interactions
- Uses LLM to compress information
- Maintains essence, not details
- Summary evolves over time

**Use Cases**:
- Long-running conversations
- Therapy/coaching chatbots
- Multi-session interactions

**Pros**:
- Handles unlimited conversation length
- Preserves key information
- Bounded token usage

**Cons**:
- Information loss in summarization
- Additional LLM calls (cost/latency)
- Summary quality varies

**Implementation**:
```python
from langchain.memory import ConversationSummaryMemory
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
memory = ConversationSummaryMemory(llm=llm)
```

**Summary Evolution**:
```
Turn 1-2: "User introduced themselves as Alice, interested in Python."
Turn 3-5: "Alice asked about data structures. Discussed lists and dicts."
Turn 6-10: "Deep dive into async programming. Alice building web scraper."
```

---

### 5. ConversationSummaryBufferMemory

**Description**: Hybrid approach - keeps recent messages verbatim and summarizes older ones.

**Characteristics**:
- Two-tier memory system
- Recent messages: full detail
- Older messages: summarized
- Best of both worlds

**Use Cases**:
- Extended conversations
- Context-sensitive applications
- Most general-purpose scenarios

**Pros**:
- Balances detail and efficiency
- Retains recent nuance
- Handles long conversations
- Most flexible option

**Cons**:
- More complex configuration
- Still requires summarization
- Additional LLM calls

**Implementation**:
```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI

llm = OpenAI()
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=500  # Recent messages stay until this limit
)
```

**Memory Structure**:
```
[Summary of old messages] + [Recent message 1] + [Recent message 2] + ...
```

---

### 6. ConversationKGMemory (Knowledge Graph Memory)

**Description**: Extracts entities and relationships from conversations into a knowledge graph.

**Characteristics**:
- Builds structured knowledge
- Captures relationships between entities
- Uses graph representation
- Enables complex reasoning

**Use Cases**:
- Complex domain knowledge
- Multi-entity tracking
- Relationship-heavy conversations
- Research assistants

**Pros**:
- Structured knowledge representation
- Captures relationships
- Enables graph-based queries
- Scalable knowledge base

**Cons**:
- Complex setup
- Requires entity extraction
- Graph maintenance overhead
- May miss unstructured info

**Implementation**:
```python
from langchain.memory import ConversationKGMemory
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
memory = ConversationKGMemory(llm=llm)
```

**Knowledge Graph Example**:
```
Alice --[works_at]--> TechCorp
Alice --[interested_in]--> Python
Python --[used_for]--> Web Scraping
```

---

### 7. ConversationEntityMemory

**Description**: Tracks specific entities (people, places, things) and their attributes across conversations.

**Characteristics**:
- Entity-centric memory
- Maintains entity profiles
- Updates facts over time
- Uses LLM for extraction

**Use Cases**:
- Customer relationship management
- Personalized assistants
- Multi-user applications
- Fact tracking

**Pros**:
- Rich entity profiles
- Fact accumulation
- Personalization support
- Structured storage

**Cons**:
- Entity extraction overhead
- May miss context outside entities
- LLM-dependent accuracy

**Implementation**:
```python
from langchain.memory import ConversationEntityMemory
from langchain.llms import OpenAI

llm = OpenAI()
memory = ConversationEntityMemory(llm=llm)
```

**Entity Store Example**:
```json
{
  "Alice": "Software engineer at TechCorp, interested in Python and web scraping",
  "TechCorp": "Alice's employer, tech company",
  "Python": "Programming language Alice is learning"
}
```

---

## Long-Term Memory Solutions

### 1. VectorStoreRetrieverMemory

**Description**: Uses vector databases to store conversation history as embeddings for semantic retrieval.

**Characteristics**:
- Embedding-based storage
- Semantic similarity search
- Scalable to large histories
- External persistence

**Use Cases**:
- Long-term user profiles
- Knowledge bases
- Multi-session applications
- Research/document analysis

**Pros**:
- Unlimited storage capacity
- Semantic retrieval (not just keyword)
- Persistent across sessions
- Efficient retrieval

**Cons**:
- Requires vector database
- Embedding costs
- Setup complexity
- May retrieve irrelevant context

**Implementation**:
```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Initialize vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    collection_name="conversation_memory",
    embedding_function=embeddings
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Retrieve top 5 relevant memories
)

# Create memory
memory = VectorStoreRetrieverMemory(retriever=retriever)
```

**Supported Vector Stores**:
- Chroma
- Pinecone
- Weaviate
- Qdrant
- FAISS
- Milvus
- Redis
- Elasticsearch

**Retrieval Configuration**:
```python
# Similarity search
search_kwargs={"k": 5}

# MMR (Maximum Marginal Relevance) for diversity
search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}

# Similarity with score threshold
search_kwargs={"k": 5, "score_threshold": 0.8}
```

---

### 2. Database-Backed Memory

**Description**: Store conversation history in traditional databases (SQL/NoSQL).

**Characteristics**:
- Full CRUD operations
- Custom schemas
- Complex querying
- Multi-user support

**Use Cases**:
- Production applications
- Multi-tenant systems
- Audit trails
- Analytics

**Implementation (PostgreSQL)**:
```python
from langchain.memory import PostgresChatMessageHistory

message_history = PostgresChatMessageHistory(
    connection_string="postgresql://user:pass@localhost/db",
    session_id="user_123"
)

# Use with memory
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    chat_memory=message_history,
    return_messages=True
)
```

**Supported Databases**:
- PostgreSQL (via `PostgresChatMessageHistory`)
- MongoDB (via `MongoDBChatMessageHistory`)
- Redis (via `RedisChatMessageHistory`)
- DynamoDB (via `DynamoDBChatMessageHistory`)
- Cassandra (via `CassandraChatMessageHistory`)

---

### 3. File-Based Memory

**Description**: Persist conversations to local files (JSON, pickle, etc.).

**Characteristics**:
- Simple persistence
- No external dependencies
- Good for development
- Limited scalability

**Implementation**:
```python
from langchain.memory import FileChatMessageHistory

message_history = FileChatMessageHistory(
    file_path="./conversation_history.json"
)

memory = ConversationBufferMemory(
    chat_memory=message_history,
    return_messages=True
)
```

---

### 4. Zep Memory

**Description**: Specialized long-term memory service with built-in summarization and extraction.

**Characteristics**:
- Managed memory service
- Automatic summarization
- Fact extraction
- Session management

**Implementation**:
```python
from langchain.memory import ZepMemory

memory = ZepMemory(
    session_id="user_session_123",
    url="http://localhost:8000",  # Zep server
    memory_key="chat_history"
)
```

**Features**:
- Automatic fact extraction
- Progressive summarization
- Hybrid search (vector + keyword)
- Privacy controls

---

### 5. Motorhead Memory

**Description**: Managed memory service optimized for production deployments.

**Characteristics**:
- Incremental summarization
- Long-term storage
- Low latency
- Stateful sessions

**Implementation**:
```python
from langchain.memory import MotorheadMemory

memory = MotorheadMemory(
    session_id="user_123",
    url="http://localhost:8080",
    memory_key="chat_history"
)
```

---

## Memory Patterns & Architectures

### Pattern 1: Tiered Memory Architecture

**Concept**: Multiple memory layers with different retention and access patterns.

```
┌─────────────────────────────────────┐
│   Working Memory (Current Context)  │  ← Immediate conversation
├─────────────────────────────────────┤
│   Short-Term Memory (Recent K)      │  ← Last N interactions
├─────────────────────────────────────┤
│   Mid-Term Memory (Summaries)       │  ← Summarized history
├─────────────────────────────────────┤
│   Long-Term Memory (Vector Store)   │  ← Full history, retrieved
└─────────────────────────────────────┘
```

**Implementation**:
```python
class TieredMemory:
    def __init__(self):
        # Working memory: current turn
        self.working = {}

        # Short-term: last 5 turns
        self.short_term = ConversationBufferWindowMemory(k=5)

        # Long-term: vector store
        self.long_term = VectorStoreRetrieverMemory(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )

    def get_context(self, query):
        # Combine all memory tiers
        recent = self.short_term.load_memory_variables({})
        relevant = self.long_term.load_memory_variables({"prompt": query})

        return {
            "recent_context": recent,
            "relevant_history": relevant
        }
```

---

### Pattern 2: Retrieval-Augmented Memory

**Concept**: Store everything, retrieve selectively based on relevance.

```
User Query → Embedding → Vector Search → Top-K Memories → Context
                            ↓
                    [Vector Database]
                    (All Past Conversations)
```

**Use Case**: Customer support, research assistants

**Implementation**:
```python
from langchain.chains import ConversationChain
from langchain.memory import VectorStoreRetrieverMemory

# All conversations stored as embeddings
memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

chain = ConversationChain(
    llm=llm,
    memory=memory
)

# Only relevant past conversations are retrieved
response = chain.run("What did we discuss about Python?")
```

---

### Pattern 3: Hybrid Summarization + Retrieval

**Concept**: Combine summarization for continuity with retrieval for specifics.

```
Current Turn
    ↓
Summary of Session
    ↓
Retrieved Relevant Past Sessions (from Vector Store)
    ↓
LLM Context
```

**Implementation**:
```python
class HybridMemory:
    def __init__(self, llm, vectorstore):
        self.summary_memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=500
        )
        self.vector_memory = VectorStoreRetrieverMemory(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )

    def load_memory_variables(self, inputs):
        # Get current session summary
        summary = self.summary_memory.load_memory_variables({})

        # Get relevant past memories
        relevant = self.vector_memory.load_memory_variables(inputs)

        return {
            "current_session": summary,
            "relevant_history": relevant
        }
```

---

### Pattern 4: Entity-Centric Memory

**Concept**: Organize memory around entities rather than chronologically.

```
User Query → Entity Extraction → Entity Lookup → Entity Context
                                        ↓
                                  [Entity Store]
                                  {
                                    "Alice": {...},
                                    "Project X": {...}
                                  }
```

**Use Case**: CRM, personalization, multi-entity tracking

**Implementation**:
```python
from langchain.memory import ConversationEntityMemory

memory = ConversationEntityMemory(llm=llm)

# Automatically extracts and updates entities
memory.save_context(
    {"input": "Alice works at TechCorp on Project X"},
    {"output": "Got it! I'll remember that."}
)

# Entity store now contains:
# {
#   "Alice": "Works at TechCorp on Project X",
#   "TechCorp": "Alice's employer",
#   "Project X": "Project Alice is working on"
# }
```

---

### Pattern 5: Multi-Memory Composition

**Concept**: Combine different memory types for different purposes.

**Implementation**:
```python
from langchain.memory import CombinedMemory

# Combine multiple memory types
memory = CombinedMemory(memories=[
    ConversationBufferWindowMemory(k=3),          # Recent context
    ConversationEntityMemory(llm=llm),            # Entity tracking
    VectorStoreRetrieverMemory(retriever=retriever)  # Long-term retrieval
])

chain = ConversationChain(llm=llm, memory=memory)
```

---

### Pattern 6: Session-Based Memory

**Concept**: Separate memory for different sessions/users.

```
User A → Session A Memory
User B → Session B Memory
User C → Session C Memory
```

**Implementation**:
```python
from langchain.memory import RedisChatMessageHistory

class SessionMemoryManager:
    def __init__(self, redis_url):
        self.redis_url = redis_url
        self.memories = {}

    def get_memory(self, session_id):
        if session_id not in self.memories:
            message_history = RedisChatMessageHistory(
                session_id=session_id,
                url=self.redis_url
            )
            self.memories[session_id] = ConversationBufferMemory(
                chat_memory=message_history,
                return_messages=True
            )
        return self.memories[session_id]

# Usage
manager = SessionMemoryManager("redis://localhost:6379")
user_memory = manager.get_memory("user_123")
```

---

### Pattern 7: Time-Decayed Memory

**Concept**: Assign weights to memories based on recency and importance.

**Implementation**:
```python
import time
from datetime import datetime

class TimeDecayedMemory:
    def __init__(self, vectorstore, decay_rate=0.1):
        self.vectorstore = vectorstore
        self.decay_rate = decay_rate

    def add_memory(self, text, importance=1.0):
        # Store with timestamp and importance
        metadata = {
            "timestamp": time.time(),
            "importance": importance
        }
        self.vectorstore.add_texts([text], metadatas=[metadata])

    def retrieve(self, query, k=5):
        # Get candidates
        docs = self.vectorstore.similarity_search(query, k=k*2)

        # Apply time decay to scores
        current_time = time.time()
        scored_docs = []

        for doc in docs:
            age_days = (current_time - doc.metadata["timestamp"]) / 86400
            decay_factor = np.exp(-self.decay_rate * age_days)
            importance = doc.metadata.get("importance", 1.0)

            score = decay_factor * importance
            scored_docs.append((doc, score))

        # Return top-k after decay
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:k]]
```

---

## Implementation Examples

### Example 1: Simple Chatbot (Buffer Window)

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import OpenAI

# Keep last 5 exchanges
memory = ConversationBufferWindowMemory(k=5)

# Create chain
chain = ConversationChain(
    llm=OpenAI(temperature=0.7),
    memory=memory,
    verbose=True
)

# Chat
print(chain.run("Hi, I'm Alice"))
print(chain.run("What's my name?"))  # Should remember "Alice"
```

---

### Example 2: Customer Support Bot (Entity Memory + Vector Store)

```python
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory, CombinedMemory
from langchain.memory import VectorStoreRetrieverMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

llm = OpenAI()

# Entity memory for customer info
entity_memory = ConversationEntityMemory(llm=llm)

# Vector store for past tickets
vectorstore = Chroma(
    collection_name="customer_support",
    embedding_function=OpenAIEmbeddings()
)
vector_memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Combine memories
memory = CombinedMemory(memories=[entity_memory, vector_memory])

# Support bot chain
chain = ConversationChain(llm=llm, memory=memory)

# Interactions
chain.run("Hi, I'm John and I'm having issues with my account")
chain.run("I can't log in")
# Entity memory tracks "John", "account issues"
# Vector memory retrieves similar past tickets
```

---

### Example 3: Long-Running Research Assistant (Summary Buffer)

```python
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

llm = OpenAI(temperature=0)

# Keep recent messages + summarize old ones
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000  # ~750 words of recent context
)

chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Can have very long conversations
# Old messages get summarized, recent ones stay detailed
for i in range(100):
    response = chain.run(f"Tell me about topic {i}")
    # Memory grows efficiently
```

---

### Example 4: Multi-User Application (Session Management)

```python
from langchain.memory import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

llm = OpenAI()

def get_user_chain(user_id):
    # Each user gets their own memory
    message_history = RedisChatMessageHistory(
        session_id=user_id,
        url="redis://localhost:6379"
    )

    memory = ConversationBufferMemory(
        chat_memory=message_history,
        return_messages=True
    )

    return ConversationChain(llm=llm, memory=memory)

# User A's conversation
chain_a = get_user_chain("user_a")
chain_a.run("Hi, I'm Alice")

# User B's conversation (separate memory)
chain_b = get_user_chain("user_b")
chain_b.run("Hi, I'm Bob")

# Memories are isolated
```

---

### Example 5: Knowledge Base with Metadata Filtering

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    collection_name="knowledge_base",
    embedding_function=embeddings
)

# Add memories with metadata
vectorstore.add_texts(
    texts=[
        "Project X budget is $100k",
        "Project Y deadline is Dec 2025",
        "Alice is the PM for Project X"
    ],
    metadatas=[
        {"project": "X", "type": "budget"},
        {"project": "Y", "type": "deadline"},
        {"project": "X", "type": "team"}
    ]
)

# Retrieve with filtering
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"project": "X"}  # Only Project X memories
    }
)

memory = VectorStoreRetrieverMemory(retriever=retriever)
```

---

### Example 6: Custom Memory Implementation

```python
from typing import Dict, List, Any
from langchain.memory import BaseMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

class CustomPriorityMemory(BaseMemory):
    """Memory that prioritizes important messages."""

    memories: List[Dict[str, Any]] = []
    max_memories: int = 10

    @property
    def memory_variables(self) -> List[str]:
        return ["history"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Sort by importance and recency
        sorted_memories = sorted(
            self.memories,
            key=lambda x: (x["importance"], x["timestamp"]),
            reverse=True
        )[:self.max_memories]

        # Format as conversation
        history = ""
        for mem in sorted_memories:
            history += f"Human: {mem['input']}\nAI: {mem['output']}\n"

        return {"history": history}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        # Calculate importance (custom logic)
        importance = self._calculate_importance(inputs["input"], outputs["output"])

        self.memories.append({
            "input": inputs["input"],
            "output": outputs["output"],
            "importance": importance,
            "timestamp": time.time()
        })

        # Prune if needed
        if len(self.memories) > self.max_memories * 2:
            self.memories = sorted(
                self.memories,
                key=lambda x: (x["importance"], x["timestamp"]),
                reverse=True
            )[:self.max_memories]

    def _calculate_importance(self, input_text: str, output_text: str) -> float:
        # Custom importance scoring
        # E.g., longer responses, specific keywords, etc.
        importance = 0.5
        if len(output_text) > 200:
            importance += 0.3
        if any(keyword in input_text.lower() for keyword in ["important", "critical", "urgent"]):
            importance += 0.2
        return min(importance, 1.0)

    def clear(self) -> None:
        self.memories = []
```

---

## Comparison Matrix

### Quick Reference

| Memory Type | Storage | Retention | Use Case | Complexity | Cost |
|-------------|---------|-----------|----------|------------|------|
| **ConversationBufferMemory** | In-memory | Unlimited | Short chats | Low | Low |
| **ConversationBufferWindowMemory** | In-memory | Last K | Recent context | Low | Low |
| **ConversationTokenBufferMemory** | In-memory | Token limit | Token budgets | Medium | Low |
| **ConversationSummaryMemory** | In-memory | Summarized | Long chats | Medium | Medium* |
| **ConversationSummaryBufferMemory** | In-memory | Hybrid | General purpose | Medium | Medium* |
| **ConversationKGMemory** | In-memory | Graph | Relationships | High | Medium* |
| **ConversationEntityMemory** | In-memory | Entities | Personalization | Medium | Medium* |
| **VectorStoreRetrieverMemory** | Vector DB | Unlimited | Long-term | High | Medium-High** |
| **Database-Backed Memory** | SQL/NoSQL | Unlimited | Production | Medium-High | Low-Medium |

*Additional LLM calls for processing
**Embedding costs + vector DB hosting

---

### Detailed Comparison

#### By Use Case

| Use Case | Recommended Memory | Alternative |
|----------|-------------------|-------------|
| Simple chatbot | ConversationBufferWindowMemory | ConversationBufferMemory |
| Customer support | ConversationEntityMemory + Vector | ConversationSummaryBufferMemory |
| Long research sessions | ConversationSummaryBufferMemory | VectorStoreRetrieverMemory |
| Multi-user app | Database-backed (Redis/PostgreSQL) | VectorStore with session filtering |
| Knowledge base | VectorStoreRetrieverMemory | ConversationKGMemory |
| Personalization | ConversationEntityMemory | VectorStore with user metadata |
| Cost-sensitive | ConversationBufferWindowMemory | ConversationTokenBufferMemory |
| Complex relationships | ConversationKGMemory | VectorStore + Entity extraction |

---

#### By Scale

| Conversation Length | Recommended Approach |
|---------------------|----------------------|
| 1-10 turns | ConversationBufferMemory |
| 10-50 turns | ConversationBufferWindowMemory |
| 50-200 turns | ConversationSummaryBufferMemory |
| 200+ turns | VectorStoreRetrieverMemory |
| Multi-session | Database-backed + VectorStore |

---

#### Performance Characteristics

| Memory Type | Read Latency | Write Latency | Scalability |
|-------------|--------------|---------------|-------------|
| Buffer | O(1) | O(1) | Poor (unbounded) |
| Buffer Window | O(k) | O(1) | Good (bounded) |
| Summary | O(1) | O(n)* | Good |
| Summary Buffer | O(1) | O(n)* | Good |
| Entity | O(entities) | O(n)* | Good |
| KG | O(nodes+edges) | O(n)* | Medium |
| Vector Store | O(log n) | O(1) | Excellent |
| Database | O(log n) | O(1) | Excellent |

*LLM call required

---

## Best Practices

### 1. Choose the Right Memory Type

```python
# Decision tree
if conversation_length < 10:
    memory = ConversationBufferMemory()
elif need_recent_context_only:
    memory = ConversationBufferWindowMemory(k=5)
elif tracking_entities:
    memory = ConversationEntityMemory(llm=llm)
elif long_running_session:
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000)
elif multi_session_or_scale:
    memory = VectorStoreRetrieverMemory(retriever=retriever)
```

---

### 2. Implement Memory Pruning

```python
from langchain.memory import ConversationBufferMemory

class PrunedMemory(ConversationBufferMemory):
    def __init__(self, max_messages=100, **kwargs):
        super().__init__(**kwargs)
        self.max_messages = max_messages

    def save_context(self, inputs, outputs):
        super().save_context(inputs, outputs)

        # Prune if too large
        if len(self.chat_memory.messages) > self.max_messages:
            # Keep most recent
            self.chat_memory.messages = self.chat_memory.messages[-self.max_messages:]
```

---

### 3. Use Metadata for Better Retrieval

```python
from langchain.vectorstores import Chroma
from datetime import datetime

vectorstore = Chroma(embedding_function=embeddings)

# Add rich metadata
vectorstore.add_texts(
    texts=["User reported login issue"],
    metadatas=[{
        "timestamp": datetime.now().isoformat(),
        "user_id": "user_123",
        "category": "authentication",
        "severity": "high",
        "session_id": "session_456"
    }]
)

# Retrieve with filters
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {
            "category": "authentication",
            "severity": "high"
        }
    }
)
```

---

### 4. Implement Memory Consolidation

```python
class ConsolidatedMemory:
    """Periodically consolidate memories to prevent bloat."""

    def __init__(self, llm, consolidation_threshold=50):
        self.llm = llm
        self.buffer = ConversationBufferMemory()
        self.consolidated = []
        self.threshold = consolidation_threshold
        self.message_count = 0

    def save_context(self, inputs, outputs):
        self.buffer.save_context(inputs, outputs)
        self.message_count += 1

        if self.message_count >= self.threshold:
            self._consolidate()

    def _consolidate(self):
        # Summarize buffer
        from langchain.chains.summarize import load_summarize_chain

        summary_chain = load_summarize_chain(self.llm, chain_type="stuff")
        summary = summary_chain.run(self.buffer.chat_memory.messages)

        # Store summary
        self.consolidated.append({
            "summary": summary,
            "timestamp": datetime.now()
        })

        # Clear buffer
        self.buffer.clear()
        self.message_count = 0

    def load_memory_variables(self, inputs):
        # Combine consolidated summaries + current buffer
        history = "\n\n".join([c["summary"] for c in self.consolidated])
        current = self.buffer.load_memory_variables({})["history"]

        return {"history": f"{history}\n\nRecent:\n{current}"}
```

---

### 5. Monitor Memory Quality

```python
class MonitoredMemory:
    """Wrapper to track memory performance."""

    def __init__(self, base_memory):
        self.base_memory = base_memory
        self.metrics = {
            "retrievals": 0,
            "writes": 0,
            "avg_retrieval_time": 0,
            "memory_size": 0
        }

    def load_memory_variables(self, inputs):
        import time
        start = time.time()

        result = self.base_memory.load_memory_variables(inputs)

        elapsed = time.time() - start
        self.metrics["retrievals"] += 1
        self.metrics["avg_retrieval_time"] = (
            (self.metrics["avg_retrieval_time"] * (self.metrics["retrievals"] - 1) + elapsed)
            / self.metrics["retrievals"]
        )

        return result

    def save_context(self, inputs, outputs):
        self.base_memory.save_context(inputs, outputs)
        self.metrics["writes"] += 1
        self.metrics["memory_size"] = self._calculate_size()

    def get_metrics(self):
        return self.metrics

    def _calculate_size(self):
        # Implement size calculation
        pass
```

---

### 6. Implement Privacy Controls

```python
class PrivacyAwareMemory:
    """Memory with PII detection and filtering."""

    def __init__(self, base_memory, pii_detector):
        self.base_memory = base_memory
        self.pii_detector = pii_detector  # E.g., Presidio, custom regex

    def save_context(self, inputs, outputs):
        # Scrub PII before storing
        clean_input = self.pii_detector.anonymize(inputs["input"])
        clean_output = self.pii_detector.anonymize(outputs["output"])

        self.base_memory.save_context(
            {"input": clean_input},
            {"output": clean_output}
        )

    def load_memory_variables(self, inputs):
        # Load scrubbed memories
        return self.base_memory.load_memory_variables(inputs)
```

---

### 7. Optimize for Cost

```python
# Use cheaper models for summarization
from langchain.llms import OpenAI

# Expensive model for main task
main_llm = OpenAI(model="gpt-4", temperature=0.7)

# Cheaper model for memory summarization
summary_llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

memory = ConversationSummaryBufferMemory(
    llm=summary_llm,  # Use cheaper model
    max_token_limit=1000
)

chain = ConversationChain(llm=main_llm, memory=memory)
```

---

### 8. Test Memory Retrieval Quality

```python
def test_memory_retrieval(memory, test_cases):
    """Evaluate memory retrieval accuracy."""

    results = []

    for query, expected_context in test_cases:
        retrieved = memory.load_memory_variables({"prompt": query})

        # Check if expected context is in retrieved
        found = any(expected in str(retrieved) for expected in expected_context)

        results.append({
            "query": query,
            "expected": expected_context,
            "retrieved": retrieved,
            "found": found
        })

    accuracy = sum(r["found"] for r in results) / len(results)

    return {
        "accuracy": accuracy,
        "details": results
    }

# Example
test_cases = [
    ("What's my name?", ["Alice"]),
    ("What project am I working on?", ["Project X"]),
]

results = test_memory_retrieval(memory, test_cases)
print(f"Retrieval accuracy: {results['accuracy']:.2%}")
```

---

### 9. Implement Fallback Strategies

```python
class FallbackMemory:
    """Try multiple memory strategies."""

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    def load_memory_variables(self, inputs):
        try:
            result = self.primary.load_memory_variables(inputs)

            # Check if result is useful
            if not result or len(str(result)) < 10:
                raise ValueError("Insufficient context")

            return result
        except Exception as e:
            print(f"Primary memory failed: {e}, using fallback")
            return self.fallback.load_memory_variables(inputs)

    def save_context(self, inputs, outputs):
        # Save to both
        self.primary.save_context(inputs, outputs)
        self.fallback.save_context(inputs, outputs)

# Usage
memory = FallbackMemory(
    primary=VectorStoreRetrieverMemory(retriever=vector_retriever),
    fallback=ConversationBufferWindowMemory(k=5)
)
```

---

### 10. Version Your Memory

```python
class VersionedMemory:
    """Track memory versions for rollback."""

    def __init__(self, base_memory):
        self.base_memory = base_memory
        self.versions = []
        self.current_version = 0

    def save_context(self, inputs, outputs):
        # Save snapshot before modification
        import copy
        self.versions.append(copy.deepcopy(self.base_memory))

        # Update memory
        self.base_memory.save_context(inputs, outputs)
        self.current_version += 1

    def rollback(self, version=None):
        """Rollback to previous version."""
        if version is None:
            version = self.current_version - 1

        if 0 <= version < len(self.versions):
            self.base_memory = self.versions[version]
            self.current_version = version
            return True
        return False
```

---

## Advanced Techniques

### 1. Hierarchical Memory

```python
class HierarchicalMemory:
    """Multi-level memory hierarchy."""

    def __init__(self, llm, vectorstore):
        # Level 1: Immediate (current turn)
        self.immediate = {}

        # Level 2: Short-term (session)
        self.short_term = ConversationBufferWindowMemory(k=10)

        # Level 3: Working (summarized session)
        self.working = ConversationSummaryMemory(llm=llm)

        # Level 4: Long-term (all sessions)
        self.long_term = VectorStoreRetrieverMemory(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
        )

    def consolidate_up(self):
        """Move information up the hierarchy."""
        # Short-term → Working memory
        if len(self.short_term.chat_memory.messages) > 10:
            # Summarize and store
            summary = self._summarize(self.short_term)
            self.working.save_context(
                {"input": "Session summary"},
                {"output": summary}
            )
            self.short_term.clear()

    def retrieve_context(self, query, levels=["all"]):
        """Retrieve from specified memory levels."""
        context = {}

        if "immediate" in levels or "all" in levels:
            context["immediate"] = self.immediate

        if "short_term" in levels or "all" in levels:
            context["short_term"] = self.short_term.load_memory_variables({})

        if "working" in levels or "all" in levels:
            context["working"] = self.working.load_memory_variables({})

        if "long_term" in levels or "all" in levels:
            context["long_term"] = self.long_term.load_memory_variables({"prompt": query})

        return context
```

---

### 2. Adaptive Memory

```python
class AdaptiveMemory:
    """Automatically adjusts memory strategy based on usage."""

    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        self.metrics = {"avg_conversation_length": 0, "num_sessions": 0}

        # Start with simple buffer
        self.current_strategy = ConversationBufferMemory()

    def adapt(self):
        """Switch memory strategy based on patterns."""
        avg_length = self.metrics["avg_conversation_length"]

        if avg_length < 10:
            self.current_strategy = ConversationBufferMemory()
        elif avg_length < 50:
            self.current_strategy = ConversationBufferWindowMemory(k=10)
        elif avg_length < 200:
            self.current_strategy = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=1000
            )
        else:
            self.current_strategy = VectorStoreRetrieverMemory(
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5})
            )

    def save_context(self, inputs, outputs):
        self.current_strategy.save_context(inputs, outputs)
        self._update_metrics()

        # Check if we should adapt
        if self.metrics["num_sessions"] % 10 == 0:
            self.adapt()

    def _update_metrics(self):
        # Update conversation length tracking
        pass
```

---

### 3. Semantic Memory Clustering

```python
from sklearn.cluster import KMeans
import numpy as np

class ClusteredMemory:
    """Group similar memories into clusters."""

    def __init__(self, vectorstore, n_clusters=5):
        self.vectorstore = vectorstore
        self.n_clusters = n_clusters
        self.clusters = {}

    def cluster_memories(self):
        """Organize memories by semantic similarity."""
        # Get all embeddings
        all_docs = self.vectorstore.get()
        embeddings = np.array([doc.embedding for doc in all_docs])

        # Cluster
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        # Organize by cluster
        for i, label in enumerate(labels):
            if label not in self.clusters:
                self.clusters[label] = []
            self.clusters[label].append(all_docs[i])

        return self.clusters

    def retrieve_from_cluster(self, query, cluster_id):
        """Retrieve from specific semantic cluster."""
        if cluster_id in self.clusters:
            # Search within cluster only
            cluster_docs = self.clusters[cluster_id]
            # Perform similarity search within cluster
            pass
```

---

### 4. Memory Importance Weighting

```python
class ImportanceWeightedMemory:
    """Assign and use importance scores for memories."""

    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore

    def calculate_importance(self, text):
        """Use LLM to rate memory importance."""
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain

        prompt = PromptTemplate(
            input_variables=["text"],
            template="""On a scale of 1-10, how important is this information
            to remember for future conversations?

            Text: {text}

            Importance (1-10):"""
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        score = chain.run(text)

        try:
            return int(score.strip()) / 10.0
        except:
            return 0.5  # Default

    def save_context(self, inputs, outputs):
        """Save with importance score."""
        combined = f"{inputs['input']} {outputs['output']}"
        importance = self.calculate_importance(combined)

        self.vectorstore.add_texts(
            texts=[combined],
            metadatas=[{"importance": importance, "timestamp": time.time()}]
        )

    def retrieve_important(self, query, k=5, min_importance=0.5):
        """Retrieve only important memories."""
        retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": k * 2,
                "filter": {"importance": {"$gte": min_importance}}
            }
        )

        return retriever.get_relevant_documents(query)[:k]
```

---

### 5. Cross-Session Memory Transfer

```python
class CrossSessionMemory:
    """Share relevant memories across user sessions."""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def save_session_context(self, session_id, inputs, outputs):
        """Save with session ID."""
        self.vectorstore.add_texts(
            texts=[f"{inputs['input']} {outputs['output']}"],
            metadatas=[{
                "session_id": session_id,
                "timestamp": time.time(),
                "shareable": self._is_shareable(inputs, outputs)
            }]
        )

    def retrieve_cross_session(self, query, exclude_session_id=None, k=5):
        """Retrieve from other sessions."""
        retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": k,
                "filter": {
                    "shareable": True,
                    "session_id": {"$ne": exclude_session_id}
                }
            }
        )

        return retriever.get_relevant_documents(query)

    def _is_shareable(self, inputs, outputs):
        """Determine if memory can be shared across sessions."""
        # Check for PII, session-specific info
        private_patterns = ["password", "token", "secret", "api key"]
        combined = f"{inputs} {outputs}".lower()

        return not any(pattern in combined for pattern in private_patterns)
```

---

### 6. Memory Validation & Correction

```python
class ValidatedMemory:
    """Validate and correct memories over time."""

    def __init__(self, llm, base_memory):
        self.llm = llm
        self.base_memory = base_memory
        self.corrections = []

    def validate_fact(self, fact):
        """Use LLM to validate a stored fact."""
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain

        prompt = PromptTemplate(
            input_variables=["fact", "history"],
            template="""Given the conversation history, is this fact still accurate?

            Fact: {fact}

            History: {history}

            Answer (Yes/No/Uncertain): """
        )

        history = self.base_memory.load_memory_variables({})
        chain = LLMChain(llm=self.llm, prompt=prompt)

        result = chain.run(fact=fact, history=history)
        return result.strip().lower()

    def correct_memory(self, old_fact, new_fact):
        """Update a stored fact."""
        self.corrections.append({
            "old": old_fact,
            "new": new_fact,
            "timestamp": time.time()
        })

        # Update in vector store (if using)
        # self.vectorstore.delete(old_fact)
        # self.vectorstore.add_texts([new_fact])

    def periodic_validation(self):
        """Regularly validate stored facts."""
        # Retrieve all memories
        # Validate each
        # Correct if needed
        pass
```

---

### 7. Multi-Modal Memory

```python
class MultiModalMemory:
    """Handle text, images, audio in memory."""

    def __init__(self):
        self.text_memory = ConversationBufferMemory()
        self.image_store = {}  # image_id -> image_data
        self.audio_store = {}  # audio_id -> audio_data
        self.references = []   # Link text to media

    def save_context(self, inputs, outputs, media=None):
        """Save with optional media attachments."""
        # Save text
        self.text_memory.save_context(inputs, outputs)

        # Save media if present
        if media:
            if media["type"] == "image":
                image_id = self._generate_id()
                self.image_store[image_id] = media["data"]

                # Link to conversation
                self.references.append({
                    "conversation_id": len(self.text_memory.chat_memory.messages),
                    "media_type": "image",
                    "media_id": image_id,
                    "timestamp": time.time()
                })

            elif media["type"] == "audio":
                audio_id = self._generate_id()
                self.audio_store[audio_id] = media["data"]

                self.references.append({
                    "conversation_id": len(self.text_memory.chat_memory.messages),
                    "media_type": "audio",
                    "media_id": audio_id,
                    "timestamp": time.time()
                })

    def load_memory_variables(self, inputs):
        """Load text and associated media."""
        text_memory = self.text_memory.load_memory_variables(inputs)

        # Get associated media
        recent_refs = self.references[-5:]  # Last 5 media items
        media = []

        for ref in recent_refs:
            if ref["media_type"] == "image":
                media.append({
                    "type": "image",
                    "data": self.image_store[ref["media_id"]]
                })
            elif ref["media_type"] == "audio":
                media.append({
                    "type": "audio",
                    "data": self.audio_store[ref["media_id"]]
                })

        return {
            "text": text_memory,
            "media": media
        }

    def _generate_id(self):
        import uuid
        return str(uuid.uuid4())
```

---

### 8. Memory Debugging Tools

```python
class DebugMemory:
    """Wrapper for debugging memory behavior."""

    def __init__(self, base_memory, log_file="memory_debug.log"):
        self.base_memory = base_memory
        self.log_file = log_file
        self.operation_count = 0

    def load_memory_variables(self, inputs):
        self.operation_count += 1

        result = self.base_memory.load_memory_variables(inputs)

        self._log({
            "operation": "load",
            "operation_id": self.operation_count,
            "inputs": inputs,
            "result": result,
            "timestamp": time.time()
        })

        return result

    def save_context(self, inputs, outputs):
        self.operation_count += 1

        self.base_memory.save_context(inputs, outputs)

        self._log({
            "operation": "save",
            "operation_id": self.operation_count,
            "inputs": inputs,
            "outputs": outputs,
            "timestamp": time.time()
        })

    def _log(self, entry):
        import json
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def analyze_logs(self):
        """Analyze logged operations."""
        import json

        with open(self.log_file, "r") as f:
            logs = [json.loads(line) for line in f]

        return {
            "total_operations": len(logs),
            "loads": sum(1 for log in logs if log["operation"] == "load"),
            "saves": sum(1 for log in logs if log["operation"] == "save"),
            "avg_retrieval_size": np.mean([len(str(log.get("result", ""))) for log in logs if log["operation"] == "load"])
        }
```

---

## Summary & Decision Guide

### Quick Start Guide

**1. For simple chatbots (< 10 turns):**
```python
memory = ConversationBufferMemory()
```

**2. For moderate conversations (10-50 turns):**
```python
memory = ConversationBufferWindowMemory(k=10)
```

**3. For long conversations (50+ turns):**
```python
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000)
```

**4. For production applications:**
```python
# Multi-user with persistence
message_history = RedisChatMessageHistory(session_id=user_id, url=redis_url)
memory = ConversationBufferMemory(chat_memory=message_history)
```

**5. For knowledge-intensive applications:**
```python
# Vector store for semantic retrieval
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
memory = VectorStoreRetrieverMemory(retriever=retriever)
```

---

### Key Takeaways

1. **No one-size-fits-all**: Choose based on your specific needs
2. **Start simple**: Begin with buffer memory, upgrade as needed
3. **Combine strategies**: Use multiple memory types together
4. **Monitor performance**: Track retrieval quality and costs
5. **Plan for scale**: Use external storage (vector stores, databases)
6. **Implement pruning**: Prevent unbounded growth
7. **Test retrieval**: Ensure relevant context is retrieved
8. **Consider privacy**: Scrub PII before storage
9. **Optimize costs**: Use cheaper models for summarization
10. **Version control**: Track memory changes for debugging

---

## Resources

### Official Documentation
- [LangChain Memory Docs](https://python.langchain.com/docs/modules/memory/)
- [LangChain Memory Types](https://python.langchain.com/docs/modules/memory/types/)

### Vector Stores
- Chroma: https://www.trychroma.com/
- Pinecone: https://www.pinecone.io/
- Weaviate: https://weaviate.io/
- Qdrant: https://qdrant.tech/

### Related Tools
- Zep: https://www.getzep.com/
- Motorhead: https://github.com/getmetal/motorhead

---

**Last Updated**: 2025-12-30
**LangChain Version**: 0.1.x+
**Author**: Claude (Anthropic)
