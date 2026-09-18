# Text2Cypher: Fine-Tuning Llama-3.2-1B with QLoRA & AST Validation

An end-to-end implementation for fine-tuning an open-weights Small Language Model (SLM) to convert natural language queries into executable Neo4j Cypher queries.

Crucially, this project moves beyond standard lexical evaluation (like BLEU or ROUGE) and implements **Abstract Syntax Tree (AST) Parsing** to strictly validate the structural correctness of the generated graph queries.

---

## 📌 Project Overview

Translating natural language questions into database queries is a foundational task for knowledge graphs and retrieval systems. While Text-to-SQL is common, graph databases like **Neo4j** rely on **Cypher**, a declarative query language with pattern-matching syntax:

```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE m.title = 'The Matrix'
RETURN p.name

```

Off-the-shelf generalist models often struggle to consistently ground queries in exact schema definitions or output pure, executable code. This project explores fine-tuning **Llama-3.2-1B-Instruct** using parameter-efficient fine-tuning (PEFT/QLoRA) to create a specialized, lightweight Cypher generation engine.

---

## 📊 Evaluation: The "Token F1 Trap" & AST Validation

Standard LLM evaluation metrics (Token Precision/Recall, F1-Score, Sequence Similarity) are fundamentally flawed for code generation. A generated query can achieve a high F1 score by memorizing database labels, but completely fail to execute due to a single misplaced bracket or invalid relationship arrow.

To accurately measure model performance, this project employs a custom evaluation metric using **PyCypher**. Instead of just counting matching tokens, the metric attempts to parse the generated text into a Cypher Abstract Syntax Tree (AST).

**The Custom Metric Pipeline:**

1. **Lexical Overlap (F1 & Gestalt Pattern Matching):** Measures how closely the generated vocabulary matches the ground truth.
2. **Offline Syntax Validation (`pycypher`):** Definitively tests if the model generated mathematically valid Cypher grammar. If the AST parser catches an error, the query is marked invalid regardless of its F1 score.

This strict validation prevents the deployment of models that generate structurally broken queries masked by high token-overlap scores.

---

## 🛠️ Stack & Technologies Used

* **Base Model:** `meta-llama/Llama-3.2-1B-Instruct`
* **Dataset:** `neo4j/text2cypher-2024v1`
* **Evaluation:** `pycypher` (Offline AST parsing and syntax validation)
* **Fine-Tuning Framework:**
* **Hugging Face `transformers` & `trl`:** Orchestrated with `SFTTrainer` and `SFTConfig`.
* **`peft` (QLoRA):** Low-Rank Adaptation (LoRA) targeted across projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **`bitsandbytes`:** 4-bit NormalFloat (NF4) quantization with double quantization.


* **Environment:** Kaggle / Google Colab (NVIDIA Tesla T4 GPU).

---

## ⚙️ Resource-Constrained Implementation Details

This iteration was deliberately engineered to train within strict memory and session-time limits (NVIDIA T4 16GB VRAM):

| Parameter | Value | Constraint Rationale |
| --- | --- | --- |
| **Quantization** | 4-bit (NF4) | Reduces model memory footprint to under 2GB. |
| **Training Subset** | 3,000 samples | Downsampled from ~40,000 to fit session execution limits. |
| **Evaluation Subset** | 300 samples | Reduced evaluation overhead. |
| **Epochs** | 1 | Kept to the minimum to complete execution in one run. |
| **Sequence Packing** | `packing=True` | Packed sequences to minimize token padding. |

---

## ⚠️ Known Gotchas, Hallucinations & Failure Modes

During testing under the resource constraints listed above, the fine-tuned adapter exhibited structural degradation caught by our AST parser.

### 1. Sample Query Comparison

* **Prompt:** *"Which Toyota vehicles have overheating problems?"*
* **Base Model Output (AST Valid):**
```cypher
MATCH (v:Vehicle { make: 'Toyota' })-[:HAS_PROBLEM]->(p:Problem)
WHERE p.name = 'Overheating'
RETURN v.name AS VehicleName, p.name AS ProblemName

```


* **Fine-Tuned Model Output (AST Invalid):**
```cypher
MATCH (v:Vehicle) WHERE v.make = 'Toyota' AND v.year < 2000 AND v.year > 1995 AND HAS_PROBLEM {problem: 'overheating'} RETURN v.name

```



### 2. Root Cause Analysis of the Faults

1. **Syntax Degradation Detected by PyCypher:**
* *The Fault:* The fine-tuned model replaced graph edge traversal syntax (`-[:HAS_PROBLEM]->`) with an invalid predicate expression (`AND HAS_PROBLEM {problem: 'overheating'}`).
* *The Metric Reality:* While the Token F1 score roughly doubled, the AST parser correctly flagged this as an unexecutable query. A 1B-parameter model requires multiple training passes (3–5 epochs) to cement strict domain grammar.


## 💡 Recommendations for Full-Scale Runs

If reproducing this pipeline on higher-tier compute (e.g., NVIDIA A100, L4, or RTX 4090):
1. **Increase Epochs:** Train for **3 to 5 epochs** to allow the model to fully converge on Cypher query syntax.
2. **Use the Full Dataset:** Remove the subset constraints and train across the entire ~40k training set.
3. **Completion-Only Masking:** Use `DataCollatorForCompletionOnlyLM` to compute loss exclusively on the Cypher response rather than penalizing the model on schema tokens.
4. **Select Best Checkpoint by AST:** Configure the `SFTTrainer` to select the best model checkpoint based on the `Syntax Valid` metric rather than raw validation loss.
