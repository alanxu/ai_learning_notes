# STORM Architecture Deep Dive

A comprehensive analysis of STORM (Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking) - Stanford's LLM-powered system for generating Wikipedia-like articles from scratch.

## Overview

**STORM** is a knowledge curation engine that automatically researches topics and generates comprehensive, well-cited articles similar to Wikipedia entries.

**Key Innovation:** Multi-perspective question asking - the system simulates conversations between a Wikipedia writer and topic experts from different perspectives, enabling deeper and broader information coverage.

**Package:** `pip install knowledge-storm`
**Repository:** https://github.com/stanford-oval/storm
**Paper:** NAACL 2024

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STORM Pipeline                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    STAGE 1: Pre-Writing                               │   │
│  │                  (Knowledge Curation)                                 │   │
│  │                                                                       │   │
│  │  ┌─────────────┐    ┌──────────────────────────────────────────┐     │   │
│  │  │   Persona   │───►│     Multi-Perspective Conversations      │     │   │
│  │  │  Generator  │    │                                          │     │   │
│  │  └─────────────┘    │  ┌────────────┐     ┌────────────────┐  │     │   │
│  │                      │  │ WikiWriter │◄───►│  TopicExpert   │  │     │   │
│  │                      │  │ (Questions)│     │ (RAG Answers)  │  │     │   │
│  │                      │  └────────────┘     └───────┬────────┘  │     │   │
│  │                      │                             │           │     │   │
│  │                      │                      ┌──────▼──────┐    │     │   │
│  │                      │                      │  Retriever  │    │     │   │
│  │                      │                      │ (Web Search)│    │     │   │
│  │                      │                      └─────────────┘    │     │   │
│  │                      └──────────────────────────────────────────┘     │   │
│  │                                      │                                │   │
│  │                                      ▼                                │   │
│  │                        ┌─────────────────────────┐                    │   │
│  │                        │ StormInformationTable   │                    │   │
│  │                        │ (Collected References)  │                    │   │
│  │                        └─────────────────────────┘                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    STAGE 2: Writing                                   │   │
│  │                                                                       │   │
│  │  ┌─────────────────┐   ┌─────────────────┐   ┌──────────────────┐    │   │
│  │  │     Outline     │──►│     Article     │──►│     Article      │    │   │
│  │  │   Generation    │   │   Generation    │   │    Polishing     │    │   │
│  │  └─────────────────┘   └─────────────────┘   └──────────────────┘    │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│                                      ▼                                       │
│                       ┌─────────────────────────────┐                        │
│                       │   Wikipedia-style Article   │                        │
│                       │      with Citations         │                        │
│                       └─────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. STORMWikiRunner (Main Orchestrator)

Located in: `knowledge_storm/storm_wiki/engine.py`

The main entry point that coordinates the entire pipeline:

```python
from knowledge_storm import STORMWikiRunner, STORMWikiRunnerArguments, STORMWikiLMConfigs
from knowledge_storm.rm import YouRM

# Configure LLMs for different tasks
lm_configs = STORMWikiLMConfigs()
lm_configs.set_conv_simulator_lm(gpt_35_model)      # Cheaper: conversations
lm_configs.set_question_asker_lm(gpt_35_model)      # Cheaper: questions
lm_configs.set_outline_gen_lm(gpt_4_model)          # Powerful: outlines
lm_configs.set_article_gen_lm(gpt_4_model)          # Powerful: writing
lm_configs.set_article_polish_lm(gpt_4_model)       # Powerful: polishing

# Configure runner arguments
args = STORMWikiRunnerArguments(
    output_dir="./output",
    max_conv_turn=3,           # Max questions per conversation
    max_perspective=3,         # Number of personas
    max_search_queries_per_turn=3,
    search_top_k=3,
    retrieve_top_k=3,
    max_thread_num=10
)

# Initialize with retriever
runner = STORMWikiRunner(args, lm_configs, rm=YouRM(api_key="..."))

# Run the full pipeline
runner.run(
    topic="Artificial Intelligence",
    do_research=True,
    do_generate_outline=True,
    do_generate_article=True,
    do_polish_article=True
)
```

### 2. LM Configuration Strategy

STORM uses different LLMs for different complexity levels:

| Task | LM Config | Recommended Model | Purpose |
|------|-----------|-------------------|---------|
| `conv_simulator_lm` | Cheaper/Faster | GPT-4o-mini | Query generation, answers |
| `question_asker_lm` | Cheaper/Faster | GPT-4o-mini | Generate questions |
| `outline_gen_lm` | More Powerful | GPT-4o | Create article structure |
| `article_gen_lm` | More Powerful | GPT-4o | Write section content |
| `article_polish_lm` | Most Powerful | GPT-4o | Final polish & summary |

---

## Stage 1: Knowledge Curation

### The Multi-Perspective Approach

The key innovation in STORM is simulating multiple expert perspectives:

```
Topic: "Quantum Computing"

Persona 1: "Computer Scientist focusing on algorithms"
Persona 2: "Physicist specializing in quantum mechanics"
Persona 3: "Industry analyst tracking commercial applications"
Persona 4: "Basic fact writer" (always included)
```

Each persona conducts a separate conversation, asking questions from their unique viewpoint.

### Persona Generator

Located in: `knowledge_storm/storm_wiki/modules/persona_generator.py`

```python
class StormPersonaGenerator:
    """Discovers perspectives by analyzing related Wikipedia articles."""

    def generate_personas(self, topic: str, max_perspective: int) -> List[str]:
        # 1. Find related Wikipedia articles
        related_topics = self.find_related_topic(topic)

        # 2. Analyze their outlines to identify perspectives
        personas = self.generate_persona(topic, related_topics)

        # 3. Always include a basic fact writer
        personas.append("Basic fact writer focusing on widely accepted facts")

        return personas[:max_perspective]
```

### Conversation Simulator

Located in: `knowledge_storm/storm_wiki/modules/knowledge_curation.py`

The `ConvSimulator` orchestrates conversations between `WikiWriter` and `TopicExpert`:

```python
class ConvSimulator(dspy.Module):
    """Simulate conversation between Wikipedia writer and expert."""

    def forward(self, topic: str, persona: str) -> List[DialogueTurn]:
        dlg_history = []

        for _ in range(self.max_turn):
            # WikiWriter asks a question based on persona
            question = self.wiki_writer(
                topic=topic,
                persona=persona,
                dialogue_turns=dlg_history
            ).question

            # Stop if writer says thank you
            if question.startswith("Thank you so much for your help!"):
                break

            # TopicExpert answers using RAG
            expert_output = self.topic_expert(
                topic=topic,
                question=question
            )

            # Store the dialogue turn
            dlg_history.append(DialogueTurn(
                user_utterance=question,
                agent_utterance=expert_output.answer,
                search_queries=expert_output.queries,
                search_results=expert_output.searched_results
            ))

        return dlg_history
```

### WikiWriter (Question Asker)

Uses DSPy signatures to generate perspective-guided questions:

```python
class AskQuestionWithPersona(dspy.Signature):
    """You are an experienced Wikipedia writer with a specific focus.
    Ask good questions to get useful information.
    When done, say "Thank you so much for your help!" to end."""

    topic = dspy.InputField(prefix="Topic you want to write: ")
    persona = dspy.InputField(prefix="Your persona: ")
    conv = dspy.InputField(prefix="Conversation history:\n")
    question = dspy.OutputField()
```

### TopicExpert (RAG Answer Generator)

The expert answers questions using retrieval-augmented generation:

```python
class TopicExpert(dspy.Module):
    """Answer questions using search + LLM generation."""

    def forward(self, topic: str, question: str):
        # 1. Convert question to search queries
        queries = self.generate_queries(topic=topic, question=question)
        # Example: "What are quantum algorithms?" →
        #          ["quantum algorithms", "Shor's algorithm", "quantum speedup"]

        # 2. Search the web
        search_results = self.retriever.retrieve(queries)

        # 3. Filter unreliable sources
        filtered_results = self.filter_sources(search_results)

        # 4. Generate grounded answer
        info = self.format_search_results(filtered_results)
        answer = self.answer_question(
            topic=topic,
            question=question,
            info=info
        )

        return Prediction(
            answer=answer,
            queries=queries,
            searched_results=filtered_results
        )
```

### StormInformationTable

The consolidated storage for all collected information:

```python
class StormInformationTable:
    """Container for all research data."""

    conversations: List[Tuple[str, List[DialogueTurn]]]  # (persona, turns)
    url_to_info: Dict[str, Information]  # Deduplicated references

    def retrieve_information(self, query: str, top_k: int) -> List[Information]:
        """Semantic search over collected snippets."""
        # Encode query using sentence-transformers
        query_embedding = self.encoder.encode(query)

        # Find most similar snippets
        similarities = cosine_similarity(query_embedding, self.snippet_embeddings)
        top_indices = np.argsort(similarities)[-top_k:]

        return [self.snippets[i] for i in top_indices]
```

---

## Stage 2: Article Generation

### Outline Generation

Located in: `knowledge_storm/storm_wiki/modules/outline_generation.py`

Converts collected conversations into a hierarchical article structure:

```python
class StormOutlineGenerationModule:
    def generate_outline(self, topic: str, information_table: StormInformationTable):
        # Concatenate all dialogue turns from all perspectives
        all_conversations = self.format_conversations(information_table)

        # LLM generates outline structure
        outline = self.write_outline(
            topic=topic,
            conversations=all_conversations
        )

        return StormArticle(outline)  # Tree of ArticleSectionNode
```

**Example Output:**
```
Quantum Computing
├── Introduction
├── History
│   ├── Theoretical Foundations
│   └── Modern Developments
├── How It Works
│   ├── Quantum Bits (Qubits)
│   └── Quantum Gates
├── Applications
│   ├── Cryptography
│   └── Drug Discovery
└── Challenges and Future
```

### Article Generation

Located in: `knowledge_storm/storm_wiki/modules/article_generation.py`

Populates each section with content using semantic retrieval:

```python
class StormArticleGenerationModule:
    def generate_article(self, topic, information_table, article_with_outline):
        # Process each section (parallelized)
        with ThreadPoolExecutor(max_workers=self.max_thread_num) as executor:
            futures = []
            for section in article_with_outline.sections:
                future = executor.submit(
                    self.generate_section,
                    section,
                    information_table
                )
                futures.append(future)

            # Collect results
            for future in as_completed(futures):
                section_content = future.result()
                # Update article tree

    def generate_section(self, section, information_table):
        # 1. Semantic search for relevant snippets
        relevant_info = information_table.retrieve_information(
            query=section.title,
            top_k=self.retrieve_top_k
        )

        # 2. Generate section content with citations
        content = self.section_writer(
            section_title=section.title,
            section_context=section.context,
            collected_info=relevant_info
        )

        return content  # Includes [1], [2] style citations
```

### Article Polishing

Located in: `knowledge_storm/storm_wiki/modules/article_polish.py`

Final refinement:
1. Add summary/abstract section
2. Remove duplicate content
3. Ensure citation consistency
4. Format references

---

## Retrieval System

### Supported Retrievers

STORM supports multiple search backends in `knowledge_storm/rm.py`:

| Retriever | API | Description |
|-----------|-----|-------------|
| `YouRM` | You.com | Recommended, good quality |
| `BingSearch` | Bing API | Microsoft search |
| `GoogleSearch` | Google CSE | Google Custom Search |
| `DuckDuckGoSearchRM` | DuckDuckGo | No API key needed |
| `SerperRM` | Serper.dev | Google results via API |
| `BraveRM` | Brave Search | Privacy-focused |
| `TavilySearchRM` | Tavily | AI-optimized search |
| `VectorRM` | Qdrant | Custom document search |
| `AzureAISearch` | Azure | Enterprise search |

### Retriever Interface

```python
class Retriever:
    """Unified retrieval interface."""

    def retrieve(
        self,
        queries: List[str],
        exclude_urls: List[str] = []
    ) -> List[Information]:
        """
        Args:
            queries: Search queries to execute
            exclude_urls: URLs to filter out (e.g., ground truth)

        Returns:
            List of Information objects with:
            - url: Source URL
            - title: Page title
            - description: Meta description
            - snippets: Relevant text excerpts
        """
```

---

## DSPy Integration

STORM uses [DSPy](https://github.com/stanfordnlp/dspy) for structured LLM prompting:

### Signatures

Signatures define input/output schemas:

```python
class QuestionToQuery(dspy.Signature):
    """Convert a question to search queries."""

    topic = dspy.InputField(prefix="Topic: ")
    question = dspy.InputField(prefix="Question: ")
    queries = dspy.OutputField(prefix="Search queries:\n- ")

class AnswerQuestion(dspy.Signature):
    """Generate grounded answer from retrieved info."""

    topic = dspy.InputField(prefix="Topic: ")
    question = dspy.InputField(prefix="Question: ")
    info = dspy.InputField(prefix="Gathered information:\n")
    answer = dspy.OutputField(prefix="Response: ")
```

### Modules

Modules compose signatures with logic:

```python
class TopicExpert(dspy.Module):
    def __init__(self):
        self.generate_queries = dspy.Predict(QuestionToQuery)
        self.answer_question = dspy.Predict(AnswerQuestion)

    def forward(self, topic, question):
        queries = self.generate_queries(topic=topic, question=question)
        # ... retrieval ...
        answer = self.answer_question(topic=topic, question=question, info=info)
        return answer
```

---

## Data Flow Example

**Topic: "CRISPR Gene Editing"**

```
1. PERSONA GENERATION
   ├─ Analyze related Wikipedia articles (Gene therapy, Genetic engineering)
   └─ Generate: ["Molecular biologist", "Bioethicist", "Medical researcher", "Basic fact writer"]

2. KNOWLEDGE CURATION (Parallel per persona)

   Persona: "Molecular biologist"
   ├─ Turn 1:
   │   ├─ WikiWriter: "How does the CRISPR-Cas9 mechanism work at the molecular level?"
   │   ├─ QuestionToQuery: ["CRISPR Cas9 mechanism", "guide RNA structure", "DNA cleavage"]
   │   ├─ Web Search: → [Nature article, MIT review, NIH page, ...]
   │   └─ TopicExpert: "CRISPR-Cas9 works by using a guide RNA to direct the Cas9 protein..."
   │
   ├─ Turn 2:
   │   ├─ WikiWriter: "What are the different Cas proteins beyond Cas9?"
   │   └─ ... (search, answer, store)
   │
   └─ Turn 3: "Thank you so much for your help!" → End

3. CONSOLIDATION
   └─ StormInformationTable with 50+ collected references

4. OUTLINE GENERATION
   CRISPR Gene Editing
   ├─ Introduction
   ├─ Discovery and History
   │   ├─ Natural CRISPR Systems
   │   └─ Development as Tool
   ├─ Mechanism
   │   ├─ Guide RNA
   │   ├─ Cas Proteins
   │   └─ DNA Repair
   ├─ Applications
   │   ├─ Medical Treatments
   │   ├─ Agriculture
   │   └─ Research
   ├─ Ethics and Controversies
   └─ Future Directions

5. ARTICLE GENERATION (Parallel per section)
   For "Guide RNA" section:
   ├─ Semantic search: Find snippets about guide RNA
   ├─ LLM writes: "Guide RNA (gRNA) is a short synthetic RNA... [1]"
   └─ Citations link to source URLs

6. POLISHING
   ├─ Add summary paragraph
   ├─ Remove redundant content
   └─ Format references

OUTPUT: 3000+ word article with 20+ citations
```

---

## Key Design Decisions

### 1. Two-Stage Retrieval

- **Stage 1 (Curation):** Search-based - query web search APIs
- **Stage 2 (Writing):** Semantic-based - similarity search over collected info

This avoids repeated web searches during writing and enables better section-info matching.

### 2. Multi-Perspective Conversations

Benefits over single-perspective:
- **Breadth:** Different experts ask different questions
- **Depth:** Follow-up questions explore topics thoroughly
- **Balance:** Multiple viewpoints prevent bias

### 3. Parallel Execution

```python
# Conversations run in parallel
with ThreadPoolExecutor(max_workers=max_thread_num) as executor:
    futures = [
        executor.submit(conv_simulator.forward, topic, persona)
        for persona in personas
    ]
```

### 4. Source Filtering

Uses Wikipedia's reliability standards to filter out:
- Self-published sources
- Known unreliable domains
- Duplicate content

### 5. Modular LLM Configuration

Different models for different tasks optimizes cost vs. quality:
- Cheap models for high-volume tasks (query generation)
- Expensive models for quality-critical tasks (article writing)

---

## Output Files

After running, STORM generates:

```
output/
└── Artificial_Intelligence/
    ├── conversation_log.json      # All dialogues
    ├── raw_search_results.json    # Retrieved documents
    ├── direct_gen_outline.txt     # Initial outline
    ├── storm_gen_outline.txt      # Refined outline
    ├── storm_gen_article.txt      # Draft article
    ├── storm_gen_article_polished.txt  # Final article
    ├── url_to_info.json           # Reference mapping
    ├── run_config.json            # Configuration used
    └── llm_call_history.jsonl     # All LLM calls (for debugging)
```

---

## Usage Examples

### Basic Usage

```python
from knowledge_storm import STORMWikiRunner, STORMWikiRunnerArguments, STORMWikiLMConfigs
from knowledge_storm.lm import LitellmModel
from knowledge_storm.rm import YouRM

# Setup LLMs
lm_configs = STORMWikiLMConfigs()
gpt_4o_mini = LitellmModel(model='gpt-4o-mini', max_tokens=500)
gpt_4o = LitellmModel(model='gpt-4o', max_tokens=3000)

lm_configs.set_conv_simulator_lm(gpt_4o_mini)
lm_configs.set_question_asker_lm(gpt_4o_mini)
lm_configs.set_outline_gen_lm(gpt_4o)
lm_configs.set_article_gen_lm(gpt_4o)
lm_configs.set_article_polish_lm(gpt_4o)

# Setup retriever
rm = YouRM(ydc_api_key='your-api-key')

# Setup runner
args = STORMWikiRunnerArguments(output_dir='./results')
runner = STORMWikiRunner(args, lm_configs, rm)

# Generate article
runner.run(topic="Machine Learning")
```

### Using Different Providers

```python
# Anthropic Claude
from knowledge_storm.lm import LitellmModel

claude = LitellmModel(
    model='claude-3-5-sonnet-20241022',
    max_tokens=4000,
    api_key='your-anthropic-key'
)

# Local Ollama
ollama = LitellmModel(
    model='ollama/llama3',
    max_tokens=2000,
    api_base='http://localhost:11434'
)
```

### Custom Document Search

```python
from knowledge_storm.rm import VectorRM

# Search over your own documents
rm = VectorRM(
    collection_name='my_docs',
    embedding_model='BAAI/bge-m3',
    qdrant_host='localhost',
    qdrant_port=6333
)
```

---

## Summary

STORM is a sophisticated system that:

1. **Generates diverse perspectives** to explore topics broadly
2. **Simulates expert conversations** for deep information gathering
3. **Uses retrieval augmentation** to ground all content in sources
4. **Produces structured articles** with proper citations
5. **Supports multiple LLM providers** via LiteLLM integration

The architecture is highly modular, allowing researchers to:
- Swap retrieval backends
- Use different LLM providers
- Customize the conversation flow
- Extend with new modules (e.g., Co-STORM for collaborative writing)

This makes STORM both a practical tool for content generation and a research platform for studying knowledge synthesis with LLMs.
