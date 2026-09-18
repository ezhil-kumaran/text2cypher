from collections import Counter
import difflib
from pycypher import parse

def compute_metrics(generated, expected):
    # Normalize whitespace and lowercase
    gen_tokens = [t for t in generated.lower().split() if t.strip()]
    exp_tokens = [t for t in expected.lower().split() if t.strip()]

    # 1. Token-level F1 Score
    gen_counter = Counter(gen_tokens)
    exp_counter = Counter(exp_tokens)

    intersection = sum((gen_counter & exp_counter).values())
    precision = intersection / len(gen_tokens) if gen_tokens else 0.0
    recall = intersection / len(exp_tokens) if exp_tokens else 0.0

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 2. Sequence Matcher ratio (Gestalt Pattern Matching)
    seq_ratio = difflib.SequenceMatcher(None, " ".join(gen_tokens), " ".join(exp_tokens)).ratio()

    # 3. Offline Syntax Validation (PyCypher AST Parsing)
    syntax_valid = False
    syntax_errors = []
    try:
        # Parse the generated query into an Abstract Syntax Tree (AST)
        ast_result = parse(generated)
        syntax_errors = ast_result.get('errors', [])
        
        # If the errors list is empty, the syntax is perfectly valid
        if not syntax_errors:
            syntax_valid = True
    except Exception as e:
        # Catch catastrophic parsing failures
        syntax_errors = [str(e)]

    return {
        "Token Precision": precision,
        "Token Recall": recall,
        "Token F1-Score": f1,
        "Sequence Similarity": seq_ratio,
        "Syntax Valid": syntax_valid,
        "Syntax Error Count": len(syntax_errors)
    }

# --- Example Usage ---
# Assuming 'base_query' and 'expected_cypher' are already defined in your notebook
base_metrics = compute_metrics(base_query, expected_cypher)

print("=== BASE MODEL METRIC SCORES ===")
for metric, score in base_metrics.items():
    # Format floats to 4 decimal places, print booleans and ints directly
    if isinstance(score, float):
        print(f"{metric}: {score:.4f}")
    else:
        print(f"{metric}: {score}")