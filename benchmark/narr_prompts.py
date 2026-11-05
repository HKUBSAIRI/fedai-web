"""Judge prompts copied verbatim from the group repo: multi_agent/PROMPT/evaluation.py (SYSTEM_PROMPT, SUBSTANTIVE_ALIGNMENT)."""
SYSTEM_PROMPT = """You are an expert evaluator of simulated FOMC member responses. Your job is to evaluate a simulated response using explicit rubrics. 

Be strict and evidence-based.
Return valid JSON only.
"""


SUBSTANTIVE_ALIGNMENT = """You are evaluating whether a simulated answer correctly matches a reference answer.

You must ONLY use:
- the question
- the reference answer
- the simulated answer

---

[Question]
{question}

[Reference Answer]
{reference_answer}

[Simulated Answer]
{predicted_answer}

---

Step 1. Extract the reference answer.

1.1 Write the core position of the reference answer in ONE sentence. 
1.2 Write the core position of the simulated answer in ONE sentence.
1.3 List all the key points from the reference answer.
   - Each key point must be specific and non-overlapping.

---

Step 2. Compare the simulated answer against the reference.

2.1 Core Position Match:
Evaluate whether the core position of the simulated answer matches the reference answer. Label as:
- full_match: same conclusion/stance
- partial_match: broadly similar but weaker/conditional/different emphasis
- mismatch: different or opposite conclusion

2.2 Key Point Coverage:
For each key point from Step 1, label:
- covered: clearly present in simulated answer
- partially_covered: vaguely or implicitly mentioned
- not_covered: missing

2.3 Contradiction:
Evaluate whether the simulated answer directly contradicts the reference answer in any way. Label as:
- none: no conflict
- minor: small inconsistencies
- major: directly contradicts key point or conclusion

---

Step 3. Summarize the evaluation. Only return a JSON object with the following fields:

{{
    "core_position": {{
        "reference": "", 
        "simulated": ""
    }},
    "position_match": "", // choose from: full_match, partial_match, mismatch
    "key_points": ["", "", "", ...], // list of key points from reference answer
    "coverage_labels": [
        {{
            "key_point": "", 
            "coverage": "" // choose from: covered, partially_covered, not_covered
        }},
        ...
    ],
    "contradiction": {{
        "evidence": "", // quote specific text from simulated answer that supports the contradiction label
        "label": "" // choose from: none, minor, major
    }}
}}
"""

