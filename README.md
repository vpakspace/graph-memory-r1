# Graph-Memory-R1

GRPO-trained Memory Manager with Graph Memory (Neo4j) for long-context conversational AI.

## Architecture

```
Memory Manager (Qwen-2.5-3B + LoRA)  →  Graph Memory (Neo4j)  ←  Answer Agent (GPT-4o-mini)
          ↑                                    ↑                           ↑
     GRPO Training (trl)              Core/Semantic/Episodic       Retrieval + QA
```

**Key idea**: The Memory Manager agent **learns via GRPO** to manage a **graph-based memory** (Neo4j).
It decides ADD/UPDATE/DELETE operations on three memory types:
- **Core**: User profile and preferences (singleton)
- **Semantic**: Facts and knowledge (nodes + RELATED_TO edges)
- **Episodic**: Events and experiences (temporal sequence)

The Answer Agent (frozen GPT-4o-mini) uses the graph memory to answer questions.
The Memory Manager is rewarded based on downstream QA quality (F1).

## Stack

- **Memory Manager**: Qwen-2.5-3B + LoRA (~6GB VRAM)
- **Answer Agent**: OpenAI GPT-4o-mini (API)
- **Graph Memory**: Neo4j (reuses `temporal-kb-neo4j` container)
- **Training**: GRPO via `trl` (HuggingFace), LoRA fine-tuning
- **Benchmark**: LoCoMo dataset (152 train / 1307 test)
- **UI**: Streamlit (port 8505)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Neo4j
docker start temporal-kb-neo4j

# 3. Set API key
echo "OPENAI_API_KEY=sk-..." >> .env

# 4. Run UI
./run_streamlit.sh

# 5. Run tests
pytest tests/ -x -q
```

## Training

```bash
# Download LoCoMo dataset to data/locomo/
# Then:
python training/train.py --data data/locomo/locomo10.json --epochs 3
```

## Project Structure

```
graph-memory-r1/
├── config.py                # Pydantic Settings
├── memory/
│   ├── graph_memory.py      # GraphMemory (Neo4j CRUD + search)
│   ├── operations.py        # Parse & execute tool calls
│   └── retriever.py         # Hybrid retrieval (BM25 + vector + graph)
├── agents/
│   ├── memory_manager.py    # Qwen-2.5-3B Memory Manager
│   ├── answer_agent.py      # GPT-4o-mini Answer Agent
│   └── tools.py             # Tool schemas
├── training/
│   ├── grpo_trainer.py      # GRPO training loop
│   ├── reward.py            # 4-component reward function
│   ├── dataset.py           # LoCoMo dataset loader
│   └── train.py             # Training entry point
├── evaluation/
│   ├── metrics.py           # F1, EM, BLEU-1
│   ├── benchmark.py         # LoCoMo benchmark runner
│   └── llm_judge.py         # LLM-as-judge
├── ui/
│   ├── streamlit_app.py     # 4-tab UI (port 8505)
│   └── i18n.py              # EN/RU translations
└── tests/                   # 92 unit tests
```

## Reward Function

```
R = r_qa + 0.05 * r_compress + 0.1 * r_valid + 0.05 * r_tool
```

| Component | Description |
|-----------|-------------|
| r_qa | Token-level F1 of predicted vs gold answer |
| r_tool | Validity of generated tool calls (0-1) |
| r_compress | Memory compression efficiency |
| r_valid | Content placed in correct memory type |

## Tests

```bash
pytest tests/ -x -q  # 92 tests
```
