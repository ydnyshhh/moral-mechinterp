# moral-mechinterp

`moral-mechinterp` is a reproducible research scaffold for the behavioral stage of GT-HarmBench-style mechanistic interpretability. It evaluates Base, UT, and GAME Qwen models on two-choice moral and game-theoretic scenarios by scoring next-token preferences for `A` versus `B`, computing safe-action logit margins, and building disagreement sets that can anchor later analyses such as CKA, probes, logit lens, activation patching, and moral/cooperative direction studies. This repository focuses only on the behavioral foundation: clean JSONL inputs, deterministic A/B scoring, structured CSV/JSONL outputs, summary tables, and polished static figures.

## Layerwise Logit-Lens Margins

Layerwise logit-lens margins measure how strongly each intermediate layer's residual stream linearly supports the safe option over the harmful option. For each intermediate residual stream, we apply the model's final RMSNorm/final normalization layer before unembedding through the LM head; this is a normed logit lens, not a tuned lens. This is not a causal intervention, but a diagnostic of decision-evidence trajectories. We use it because behavioral differences between adapters are mostly small margin shifts rather than strong binary flips.

```bash
PYTHONPATH=src python scripts/03_logit_lens_margins.py --subset-csv outputs/behavior_full/subsets/top_ut_margin_shift.csv --config configs/eval.yaml --output-dir outputs/logit_lens/top_ut_margin_shift --models base,ut,game
PYTHONPATH=src python scripts/03_logit_lens_margins.py --subset-csv outputs/behavior_full/subsets/top_game_margin_shift.csv --config configs/eval.yaml --output-dir outputs/logit_lens/top_game_margin_shift --models base,ut,game
PYTHONPATH=src python scripts/03_logit_lens_margins.py --subset-csv outputs/behavior_full/subsets/ut_safe_game_harmful.csv --config configs/eval.yaml --output-dir outputs/logit_lens/ut_safe_game_harmful --models base,ut,game
PYTHONPATH=src python scripts/03_logit_lens_margins.py --subset-csv outputs/behavior_full/subsets/game_safe_ut_harmful.csv --config configs/eval.yaml --output-dir outputs/logit_lens/game_safe_ut_harmful --models base,ut,game
```

## Paper Figure

The main mechanistic framing is: aggregate safe-choice rates are flat, but reward adapters create objective-specific late-layer decision-evidence shifts. Interpret the plot as relative separation in the final third of the network, not as evidence for a single moral layer. Regenerate the combined 2x2 logit-lens figure, late-layer separation tables, and random PD/Chicken control subset CSVs with:

```bash
PYTHONPATH=src python scripts/04_make_paper_figures.py
```

## Random Game-Type Controls

The random PD/Chicken controls test whether late-layer adapter separation appears beyond margin-shift-selected or categorical-disagreement subsets. Create or validate the fixed random subsets, run the existing logit-lens script, then summarize layers 21-31 with:

```bash
PYTHONPATH=src python scripts/04_make_random_control_subsets.py

PYTHONPATH=src python scripts/03_logit_lens_margins.py \
  --subset-csv outputs/behavior_full/subsets/random_pd_150.csv \
  --config configs/eval.yaml \
  --output-dir outputs/logit_lens_fixed/random_pd_150 \
  --models base,ut,game

PYTHONPATH=src python scripts/03_logit_lens_margins.py \
  --subset-csv outputs/behavior_full/subsets/random_chicken_150.csv \
  --config configs/eval.yaml \
  --output-dir outputs/logit_lens_fixed/random_chicken_150 \
  --models base,ut,game

PYTHONPATH=src python scripts/05_summarize_control_logit_lens.py
```
