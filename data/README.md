# Data Directory

Place GT-HarmBench-style JSONL benchmark files here. The converted benchmark is
`gtharmbench.jsonl`.

Preferred schema:

```json
{"id":"pd_0001","game_type":"prisoners_dilemma","scenario":"...","option_a":"...","option_b":"...","safe_label":"A"}
```

The local source CSV is ignored by git; regenerate the JSONL with
`scripts/convert_gtharmbench_csv.py` if the source file changes.
