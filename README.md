
# Text2Cypher: Fine-Tuning Llama-3.2-1B with QLoRA

An end-to-end implementation for fine-tuning an open-weights Small Language Model (SLM) to convert natural language queries into executable Neo4j Cypher queries using graph schema definitions.

---

## 📌 Project Overview

Translating natural language questions into database queries is a foundational task for knowledge graphs and retrieval systems. While Text-to-SQL is common, graph databases like **Neo4j** rely on **Cypher**, a declarative query language with pattern-matching syntax:

```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE m.title = 'The Matrix'
RETURN p.name

```

Off-the-shelf generalist models often struggle to consistently ground queries in exact schema definitions, hallucinate non-existent edge types, or add conversational filler. This project explores fine-tuning **Llama-3.2-1B-Instruct** using parameter-efficient fine-tuning (PEFT/QLoRA) to create a specialized, lightweight Cypher generation engine.

---

## 🚀 Why This Is Useful

* **Graph-RAG Pipelines:** Serves as the query-translation engine inside Graph-Augmented Retrieval pipelines, enabling semantic search systems to dynamically pull structured facts directly from Neo4j.
* **Non-Technical Access:** Enables non-technical domain experts (analysts, healthcare staff, compliance teams) to interrogate complex connected data using plain English without writing graph queries.
* **Edge & Cost-Efficient Deployment:** By targeting a 1B-parameter model with 4-bit quantization, the resulting adapter can be served on low-cost compute instances, edge devices, or localized setups without incurring proprietary API costs.

---

## 🛠️ Stack & Technologies Used

* **Base Model:** `meta-llama/Llama-3.2-1B-Instruct`
* **Dataset:** `neo4j/text2cypher-2024v1` (official Neo4j Text2Cypher benchmark dataset containing schema metadata, natural language questions, and canonical Cypher queries)
* **Fine-Tuning Framework:**
* **Hugging Face `transformers` & `trl`:** Orchestrated with `SFTTrainer` and `SFTConfig`.
* **`peft` (QLoRA):** Low-Rank Adaptation (LoRA) targeted across projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **`bitsandbytes`:** 4-bit NormalFloat (NF4) quantization with double quantization to fit within limited GPU memory.
* **`accelerate`:** Handles hardware mapping and gradient execution.


* **Environment:** Kaggle / Google Colab (NVIDIA Tesla T4 GPU).

---

## ⚙️ Resource-Constrained Implementation Details

This iteration was deliberately engineered to train within strict memory and session-time limits (NVIDIA T4 16GB VRAM):

| Parameter | Value | Constraint Rationale |
| --- | --- | --- |
| **Quantization** | 4-bit (NF4) | Reduces model memory footprint to under 2GB |
| **Training Subset** | 3,000 samples | Downsampled from ~40,000 to fit session execution limits |
| **Evaluation Subset** | 300 samples | Reduced evaluation overhead |
| **Epochs** | 1 | Kept to minimum to complete execution in one run |
| **Batch Configuration** | Batch size 2, Accumulation 4 | Balances gradient smoothing without OOM crashes |
| **Sequence Packing** | `packing=True` | Packed sequences to minimize token padding |

---

## ⚠️ Known Gotchas, Hallucinations & Failure Modes

During testing, the fine-tuned adapter exhibited structural degradation and unexpected hallucinated filters:

### 1. Sample Query Comparison

* **Prompt:** *"Which Toyota vehicles have overheating problems?"*
* **Ground Truth Query:**
```cypher
MATCH (v:Vehicle)-[:HAS_PROBLEM]->(p:Problem {name: 'Overheating'})
WHERE v.make = 'Toyota'
RETURN v

```


* **Base Model Output:**
```cypher
MATCH (v:Vehicle { make: 'Toyota' })-[:HAS_PROBLEM]->(p:Problem)
WHERE p.name = 'Overheating'
RETURN v.name AS VehicleName, p.name AS ProblemName

```


* **Fine-Tuned Model Output (Constrained Run):**
```cypher
MATCH (v:Vehicle) WHERE v.make = 'Toyota' AND v.year < 2000 AND v.year > 1995 AND HAS_PROBLEM {problem: 'overheating'} RETURN v.name

```



### 2. Root Cause Analysis of the Faults

1. **Syntax Degradation (Relationship Pattern Breakdown):**
* *The Fault:* The fine-tuned model replaced graph edge traversal syntax (`-[:HAS_PROBLEM]->`) with an invalid predicate expression (`AND HAS_PROBLEM {problem: 'overheating'}`).
* *The Cause:* A 1B-parameter model requires multiple training passes (3–5 epochs) to cement strict domain grammar. Training for only 1 epoch resulted in partial syntax retention where the model recognized vocabulary tokens but lost the structural graph grammar.
---

## 💡 Recommendations for Full-Scale Runs

If reproducing this pipeline on higher-tier compute (e.g., NVIDIA A100, L4, or RTX 4090):

1. **Disable Sequence Packing:** Set `packing=False` (or enable `flash_attention_2` on Ampere/Ada architectures) to eliminate cross-sequence hallucination.
2. **Increase Epochs:** Train for **3 to 5 epochs** to allow the model to fully converge on Cypher query syntax.
3. **Use the Full Dataset:** Remove the `.select(range(...))` constraints and train across the entire ~40k training set.
4. **Completion-Only Masking:** Use `DataCollatorForCompletionOnlyLM` to compute loss exclusively on the Cypher response rather than penalizing the model on schema tokens.


