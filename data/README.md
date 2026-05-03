# Data Directory

Place GT-HarmBench-style JSONL benchmark files here. Use
`gtharmbench_balanced.jsonl` for evaluation; it is a seeded A/B-position-balanced
version of `gtharmbench.jsonl`.

Preferred schema:

```json
{"id":"pd_0001","game_type":"prisoners_dilemma","scenario":"...","option_a":"...","option_b":"...","safe_label":"A"}
```

The local source CSV is ignored by git. Regenerate the converted JSONL with
`scripts/convert_gtharmbench_csv.py`, then regenerate the balanced eval file
with `scripts/balance_ab_positions.py`.
