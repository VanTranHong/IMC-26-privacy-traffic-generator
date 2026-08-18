"""Shared scoring logic for data-extraction attacks.

Adapted from `calculate_cc()` in a private internal research harness's
`calculate_memorization_attack.py` (not included in this repo)
-- the original function reported two metrics it called "coverage" and
"confidence". This module reports the same underlying counts, but under the
standard **precision/recall** framing instead (see `PRECISION_RECALL_NOTE`
below for the exact correspondence), plus F1 as a single combined number.
Used by netssm_data_extraction.py, trafficllm_data_extraction.py, and
netdiffusion_data_extraction.py in this folder.

## What a "data-extraction attack" measures here

A generative model is fed prompts (or run unconditionally) to produce many
synthetic samples. For each sample, the attacker reads off one specific field
(e.g. a source IP address, a MAC address, or a structural fingerprint of a
packet) and asks: **did the model just reproduce a real value from its
training data?** If so, the model has "leaked" (extracted) training data
through its generation outputs, not through prompted memorization per se --
this is complementary to the membership-inference attacks in
../MIA (which ask "was this specific known sample a training member?"),
whereas here the attacker doesn't need candidate samples up front, just the
ability to freely generate outputs and see what comes out.

## Precision / recall, precisely

Let:
  - `original_values`  = every occurrence of the field in the real training data
                          (a list, with repeats -- frequency matters here).
  - `generated_values` = every occurrence of the field across all generated
                          samples (again, a list with repeats).
  - `overlap`          = the set of *unique* values that appear in both.

Then:
  - **precision** = (# generated occurrences whose value is in `overlap`)
                     / (# generated occurrences)
                   "Of everything the model generated, what fraction was a
                   real, memorized training value (rather than a novel,
                   synthesized one)?"
                   This is exactly the original script's "confidence".
  - **recall**    = (# unique original values that were reproduced at least
                     once anywhere in `generated_values`) / (# unique
                     original values)
                   "Of all the distinct real values in the training data,
                   what fraction did the attacker manage to recover by
                   generating enough samples?"
                   This is exactly the original script's "coverage".
  - **f1**        = harmonic mean of precision and recall, a single number
                     summarizing overall extraction success.

A high-precision, low-recall attack reliably reproduces a small set of
memorized values; a low-precision, high-recall attack reproduces most of the
training vocabulary but drowns it in synthesized noise. Both are useful to
know, which is why both are reported (rather than picking one).
"""

from collections import Counter
from typing import Dict, List


def precision_recall_extraction(original_values: List[str], generated_values: List[str]) -> Dict[str, float]:
    """Score a data-extraction attack via precision/recall (see module
    docstring for exact definitions and the correspondence with the original
    script's "confidence"/"coverage" terminology).

    Both inputs are flat lists of field values (e.g. one IP address string
    per packet), with repeats -- do not de-duplicate before calling this.
    """
    if not original_values or not generated_values:
        raise ValueError("Both original_values and generated_values must be non-empty.")

    original_unique = set(original_values)
    generated_counts = Counter(generated_values)

    overlap = original_unique & set(generated_values)
    true_positive_instances = sum(generated_counts[v] for v in overlap)

    precision = true_positive_instances / len(generated_values)
    recall = len(overlap) / len(original_unique)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "num_original_occurrences": len(original_values),
        "num_original_unique": len(original_unique),
        "num_generated_occurrences": len(generated_values),
        "num_generated_unique": len(generated_counts),
        "num_overlap_unique": len(overlap),
        "num_true_positive_instances": true_positive_instances,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_report(field_name: str, metrics: Dict[str, float]) -> None:
    """Human-readable summary of a precision_recall_extraction() result."""
    print(f"--- Data-extraction attack report: {field_name} ---")
    print(f"  Original occurrences (training data):  {metrics['num_original_occurrences']}")
    print(f"  Original unique values:                {metrics['num_original_unique']}")
    print(f"  Generated occurrences:                 {metrics['num_generated_occurrences']}")
    print(f"  Generated unique values:                {metrics['num_generated_unique']}")
    print(f"  Unique values recovered (overlap):      {metrics['num_overlap_unique']}")
    print(f"  Precision (fraction of generated output that is real training data): {metrics['precision']:.4f}")
    print(f"  Recall (fraction of training vocabulary successfully recovered):    {metrics['recall']:.4f}")
    print(f"  F1:                                                                  {metrics['f1']:.4f}")
