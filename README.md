# Graph-Memory-R1

GRPO-trained Memory Manager with Graph Memory (Neo4j) for long-context conversational AI.

Inspired by the [MemoryAgent](https://arxiv.org/abs/2502.12217) paper (Xu et al., 2025) — a reinforcement learning approach to structured memory management in LLM agents.

## Architecture

```
Conversation chunks
        │
        ▼
┌─────────────────────────┐
│  Memory Manager          │
│  (Qwen-2.5-3B + LoRA)   │──► <tool_call> ADD / UPDATE / DELETE / NOOP
│  Trained via GRPO        │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Graph Memory (Neo4j)    │
│  ┌─────┐ ┌────────┐     │
│  │Core │ │Semantic│     │   3 memory types
│  └─────┘ └────────┘     │   + relationships
│  ┌────────┐              │
│  │Episodic│              │
│  └────────┘              │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Hybrid Retriever        │
│  BM25 + Vector + Graph   │──► RRF fusion → top-K nodes
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Answer Agent            │
│  (GPT-4o-mini, frozen)   │──► Answer with confidence
└─────────────────────────┘
```

**Key idea**: The Memory Manager agent **learns via GRPO** (Group Relative Policy Optimization) to manage a **graph-based memory** stored in Neo4j.
It decides ADD / UPDATE / DELETE / NOOP operations on three memory types:

| Memory Type | Purpose | Example |
|-------------|---------|---------|
| **Core** | User profile, preferences (singleton) | "User lives in Paris" |
| **Semantic** | Facts, knowledge (nodes + RELATED_TO edges) | "Python is a programming language" |
| **Episodic** | Events, experiences (temporal sequence) | "User attended PyCon 2024" |

The Answer Agent (frozen GPT-4o-mini) retrieves relevant memory nodes and generates answers.
The Memory Manager is rewarded based on downstream QA quality (token F1).

## Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Memory Manager | Qwen-2.5-3B-Instruct + LoRA | ~6GB VRAM, LoRA rank=16, alpha=32 |
| Answer Agent | OpenAI GPT-4o-mini | Frozen, temperature=0 |
| Graph Memory | Neo4j 5.x | Reuses `temporal-kb-neo4j` container |
| Training | GRPO via `trl` 0.28+ | LoRA fine-tuning, group_size=4 |
| Embeddings | OpenAI text-embedding-3-small | 1536 dimensions |
| Dataset | LoCoMo (ACL 2024) | 10 conversations, 1986 QA pairs |
| UI | Streamlit | 4 tabs, i18n (EN/RU), port 8505 |
| CI | GitHub Actions | Python 3.11/3.12, pytest |

## Training Results

Two-stage GRPO training on RTX 4080 Laptop GPU (12GB VRAM):

### Stage 1: Structural Training
Rewards based on tool call validity and memory compression only (no Neo4j required).

| Metric | Value |
|--------|-------|
| Epochs | 3 (78 steps) |
| Duration | 6h 46m |
| Reward | 0.53 → 0.63 (+19%) |
| Loss | -0.012 |

### Stage 2: Full Reward Training
Adds QA-based reward (r_qa) — requires Neo4j for graph memory operations.

| Metric | Value |
|--------|-------|
| Epochs | 3 (78 steps) |
| Duration | 6h 58m |
| Reward | 0.07 → 0.10 (+43%) |
| Loss | 0.002 |
| LoRA resume | `PeftModel.from_pretrained(is_trainable=True)` |

**Total training time**: ~14 hours on a single RTX 4080 Laptop GPU.

The LoRA checkpoint is saved to `checkpoints/final_lora/`.

## Reward Function

```
R = r_qa + 0.05 * r_compress + 0.1 * r_valid + 0.05 * r_tool
```

| Component | Weight | Description |
|-----------|--------|-------------|
| r_qa | 1.0 | Token-level F1 between predicted and gold answer |
| r_valid | 0.1 | Content placed in correct memory type (Core/Semantic/Episodic) |
| r_compress | 0.05 | Memory compression efficiency (fewer nodes = better) |
| r_tool | 0.05 | Structural validity of generated `<tool_call>` JSON |

## Dataset: LoCoMo

[LoCoMo](https://github.com/snap-research/locomo) (Long-Context Conversation with Memory, ACL 2024) — 10 long multi-session conversations with 1986 QA pairs across 5 categories:

| Category | Count | Description |
|----------|-------|-------------|
| 1 — Single-hop | 282 | Direct factual recall |
| 2 — Multi-hop | 321 | Reasoning across multiple facts |
| 3 — Temporal | 96 | Time-sensitive questions |
| 4 — Open-domain | 841 | General knowledge integration |
| 5 — Adversarial | 446 | Tricky/misleading questions |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for Neo4j)
- NVIDIA GPU with 6+ GB VRAM (for training)
- OpenAI API key

### Setup

```bash
# 1. Clone
git clone https://github.com/vpakspace/graph-memory-r1.git
cd graph-memory-r1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Neo4j
docker start temporal-kb-neo4j
# Or create new:
# docker run -d --name temporal-kb-neo4j \
#   -p 7474:7474 -p 7687:7687 \
#   -e NEO4J_AUTH=neo4j/temporal_kb_2026 \
#   neo4j:5

# 4. Set API key
echo "OPENAI_API_KEY=sk-..." >> .env

# 5. Run tests
pytest tests/ -x -q  # 92 tests

# 6. Run UI
./run_streamlit.sh
# Opens http://localhost:8505
```

## Training

```bash
# Stage 1: Structural training (no Neo4j required)
python training/train.py \
  --data data/locomo/locomo10.json \
  --epochs 3 \
  --reward-mode structural

# Stage 2: Full reward (requires Neo4j running)
python training/train.py \
  --data data/locomo/locomo10.json \
  --epochs 3 \
  --reward-mode full \
  --resume-from checkpoints/structural_lora
```

### Inference Demo

```bash
python scripts/inference_demo.py
# Generates <tool_call> JSON operations from conversation input
```

## Benchmark

```bash
# Full benchmark (3 configs: no_memory, graph_base, graph_grpo)
python scripts/run_benchmark.py --output results.json

# Single config
python scripts/run_benchmark.py --config graph_grpo --max-qa 50

# With LLM judge
python scripts/run_benchmark.py --config graph_grpo --use-judge
```

**Configs**:
- `no_memory` — Answer Agent without graph memory (baseline)
- `graph_base` — Memory Manager (base Qwen, no LoRA) → graph → Answer Agent
- `graph_grpo` — Memory Manager (Qwen + LoRA) → graph → Answer Agent

## Streamlit UI

4-tab interface with EN/RU localization:

| Tab | Features |
|-----|----------|
| **Chat** | Process conversation chunks, ask questions with graph memory |
| **Graph Explorer** | View node/edge stats, memory snapshot, clear graph |
| **Training** | Configure and launch GRPO training, load LoRA weights |
| **Benchmark** | Run LoCoMo evaluation with progress tracking |

## Project Structure

```
graph-memory-r1/
├── config.py                    # Pydantic Settings (Neo4j, OpenAI, Qwen, Training)
├── memory/
│   ├── graph_memory.py          # GraphMemory (Neo4j CRUD + search)
│   ├── operations.py            # Parse & execute <tool_call> XML
│   └── retriever.py             # Hybrid retrieval (BM25 + vector + graph traversal, RRF)
├── agents/
│   ├── memory_manager.py        # Qwen-2.5-3B Memory Manager (load_model, load_lora, process_chunk)
│   ├── answer_agent.py          # GPT-4o-mini Answer Agent (answer, answer_with_memory)
│   └── tools.py                 # Tool schemas (add_core, add_semantic, add_episodic, ...)
├── training/
│   ├── grpo_trainer.py          # GRPO training loop (trl GRPOTrainer wrapper)
│   ├── reward.py                # 4-component reward (r_qa, r_tool, r_compress, r_valid)
│   ├── dataset.py               # LoCoMo dataset loader + conversation chunker
│   └── train.py                 # Training entry point (structural / full reward)
├── evaluation/
│   ├── metrics.py               # F1, Exact Match, BLEU-1
│   ├── benchmark.py             # Benchmark runner (per-category breakdown)
│   └── llm_judge.py             # LLM-as-judge (GPT-4o-mini, 0-100 score)
├── scripts/
│   ├── run_benchmark.py         # Full LoCoMo benchmark (3 configs, comparison table)
│   └── inference_demo.py        # Quick inference test
├── ui/
│   ├── streamlit_app.py         # 4-tab Streamlit UI (port 8505)
│   └── i18n.py                  # EN/RU translations
├── data/locomo/                 # LoCoMo dataset (10 conversations)
├── checkpoints/final_lora/      # Trained LoRA adapter
├── tests/                       # 92 unit tests
├── .github/workflows/ci.yml    # GitHub Actions CI
├── requirements.txt
└── .env                         # API keys (not committed)
```

## Tests

```bash
pytest tests/ -x -q  # 92 tests, ~0.6s
```

Tests cover all modules: config, graph_memory, operations, retriever, memory_manager, answer_agent, tools, reward, dataset, grpo_trainer, metrics, benchmark, llm_judge, i18n.

## References

- **MemoryAgent**: Xu et al., "MemoryAgent: Autonomous LLM Agent with Long-Term Memory and Self-Improving", 2025. [arXiv:2502.12217](https://arxiv.org/abs/2502.12217)
- **LoCoMo**: Maharana et al., "Evaluating Long-Context Conversational Memory", ACL 2024. [snap-research/locomo](https://github.com/snap-research/locomo)
- **GRPO**: Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- **trl**: HuggingFace TRL library for RL training. [huggingface/trl](https://github.com/huggingface/trl)

## License

MIT
