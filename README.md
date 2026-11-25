# HyPARE: Hybrid Parallel–Recurrent Neural Algorithmic Reasoning

Project repository for a study of **HyPARE**, a processor that combines
parallel MPNN message passing with node-level GRU memory through a learned gate.
The implementation follows the CLRS encode–process–decode protocol and is
evaluated on CLRS BFS and DFS.

**Members:** Member 1, Member 2, Member 3

## Results at a glance

### BFS: reproduction at the ceiling (64-node OOD, 3 seeds)

| Model | OOD accuracy |
|---|---:|
| MPNN | 99.82 ± 0.09 |
| MPNN+GRU | 99.95 ± 0.00 |
| HyPARE | **99.95 ± 0.00** |
| CLRS paper MPNN | 99.89 ± 0.05 |

### DFS: controlled improvement, not a solved benchmark (64-node OOD)

| Model | OOD accuracy |
|---|---:|
| MPNN (matched baseline) | 12.45 ± 2.66 |
| MPNN+GRU (fixed 50/50) | 9.68 ± 5.21 |
| **HyPARE** | **19.43 ± 6.09** |
| MPNN (no position aug., no LN) | 13.61 ± 4.48 |
| MPNN (no position aug., LN) | 9.34 ± 1.93 |
| CLRS paper MPNN | 6.54 ± 0.51 |
| CLRS paper PGN | 8.71 ± 0.24 |
| CLRS paper GAT | 11.78 ± 2.04 |
| CLRS paper Memnet | 13.36 ± 1.61 |
| CLRS paper Deep Sets | 7.44 ± 0.73 |

## How to read the DFS result

DFS is the interesting result, but it needs careful framing:

1. **Controlled claim.** Under identical data, training budget, and evaluation,
   HyPARE improves the matched MPNN from 12.45% to 19.43% (+6.98 points) and the
   fixed hybrid from 9.68% to 19.43% (+9.75 points).
2. **Context only.** HyPARE's 19.43% is above the best published DFS baseline in
   the CLRS paper (Memnet, 13.36%), but our MPNN baseline is also stronger than
   the paper's MPNN (12.45–13.61% versus 6.54%). Cross-paper ordering is therefore
   not a controlled comparison.
3. **Preliminary.** Per-seed HyPARE values are 18.55%, 27.29%, and 12.45%;
   matched MPNN values are 16.21%, 10.35%, and 10.79%. The per-seed gains are
   +2.34, +16.94, and +1.66 points. With three seeds we do not claim statistical
   significance.
4. **Not solved.** DFS absolute accuracy is still low and the CLRS authors
   describe DFS OOD as unsolved. The result is a promising improvement and a
   diagnosis, not a state-of-the-art claim.

## Diagnosis

The length sweep and gate analysis explain the remaining failure:

| Test nodes | MPNN | MPNN+GRU | HyPARE |
|---:|---:|---:|---:|
| 16 | 99.80 | 99.02 | 99.93 |
| 24 | 78.99 | 82.51 | 96.70 |
| 32 | 48.34 | 37.50 | 74.48 |
| 48 | 22.79 | 14.89 | 32.03 |
| 64 | 11.90 | 9.88 | 18.62 |

The graph-level gate decays over the rollout: mean values by rollout quartile are
approximately **0.41 → 0.27 → 0.12 → 0.035**. Recurrence is therefore switched
off late in the algorithm, when long-range computation is most needed. HyPARE's
advantage peaks at 24–32 nodes and narrows at 48–64 nodes. Hint accuracy is
high early and decays late, consistent with error compounding.

## Repository layout

```text
repo-root/
  src/                         CLRS adapters, models, training loops
  scripts/train_bfs.py         BFS training
  scripts/train_clrs.py        Any-algorithm training (used for DFS)
  scripts/diagnose_dfs.py      Length sweep, hint curves, gate curves
  scripts/make_presentation_assets.py
  scripts/pdf_to_pptx.py       PDF-to-PPTX export with embedded GIF
  tests/                       Pipeline and spec regression tests
  results/                     Per-seed JSON results and diagnostics
  demo/                        Live inference script, checkpoint, examples, recording guide
  presentation/                Slides (PDF/PPTX), architecture diagram, figures/tables
  report/                      Two-column report
  requirements.txt
```

## Environment

The experiments used PyTorch 2.7.1 on a single NVIDIA RTX 6000 Ada GPU. On this
machine, the working environment is:

- Python: `/home2/home/chakilam_sai/dl/claude/.venv/bin/python`
- CLRS package: `/home2/home/chakilam_sai/clrs` (must be importable)

Check the setup:

```bash
python -c "import torch, clrs; print(torch.__version__, torch.cuda.is_available())"
```

## Reproduce

BFS, three seeds:

```bash
python scripts/train_bfs.py \
  --architectures mpnn mpnn_gru hypare \
  --seeds 0 1 2 --steps 10000 --device cuda \
  --use-ln --random-pos \
  --output results/v5_bfs.json \
  --checkpoint-dir checkpoints/bfs
```

DFS, controlled comparison:

```bash
python scripts/train_clrs.py \
  --algorithm dfs \
  --architectures mpnn mpnn_gru hypare \
  --seeds 0 1 2 --steps 10000 --device cuda \
  --teacher-forcing 0.5 --use-ln --random-pos \
  --train-length 16 --val-length 16 --test-length 64 \
  --train-samples 1000 --val-samples 32 --test-samples 32 \
  --output results/v5_dfs.json \
  --checkpoint-dir checkpoints/dfs
```

DFS diagnostics (requires trained checkpoints):

```bash
python scripts/diagnose_dfs.py \
  --algorithm dfs \
  --architectures mpnn mpnn_gru hypare \
  --seeds 0 1 2 --lengths 16 24 32 48 64 \
  --samples 32 --extended-samples 256 --extended-length 64 \
  --device cuda --checkpoint-dir checkpoints/dfs \
  --output results/v5_dfs_diagnostics.json
```

Regenerate all figures and tables:

```bash
python scripts/make_presentation_assets.py
```

The result JSON files are the source of truth. `scripts/make_presentation_assets.py`
reads them and writes `presentation/summary.json`, all figures, and all LaTeX
tables. The only shipped checkpoint is the small DFS demo checkpoint under
`demo/checkpoints/`; training checkpoints are not shipped.

## Live inference demo

The demo loads the included DFS HyPARE checkpoint and animates the model rollout
step by step on a fixed CLRS graph. It begins with the ground-truth input graph
and the ground-truth DFS discovery order, then switches to the prediction.
CLRS fixes the start node and neighbour order, so this is one canonical DFS;
other valid DFS traversals also exist.

Live window:

```bash
python demo/inference_demo.py --live --delay 0.75
```

Fallback animation and still frame:

```bash
python demo/inference_demo.py --save-gif demo/inference_demo.gif \
  --save-frame demo/demo_frame.png
```

See `demo/RECORDING.md` for screen-recording instructions and the optional
64-node OOD run. `demo/make_examples.py` renders a three-example overview of
canonical DFS instances (`demo/examples_overview.png` and
`demo/examples_overview.gif`).

The PPTX contains animated GIFs on the Live Inference, More DFS Examples, and
OOD Prediction slides; play those slides in Slide Show mode.

## Artifacts

- `demo/` (live inference script, checkpoint, GIF/MP4, still frame, recording guide)
- `demo/examples_overview.png` (three canonical DFS examples)
- `demo/examples_overview.gif` (animated cycle through the three examples)
- `demo/inference_demo_ood.gif` (64-node OOD prediction, seed 0)
- `demo/ood_prediction.gif` (additional 64-node OOD prediction, seed 1)
- `demo/demo_frame_ood.png` (OOD still used on the slide)
- `presentation/` (slide deck PDF/PPTX, figures, generated tables)
- `presentation/hypare_final_deck.pptx` (embedded animated inference GIF)
- `presentation/figures/architecture.pdf` (standalone architecture diagram)
- `presentation/architecture_diagram.tex` (standalone TikZ source)
- `report/` (two-column report PDF)
- `presentation/summary.json`
- `results/v5_*.json`

## Limitations

- Three seeds only; DFS variance is large and the gain is preliminary.
- Our MPNN baseline is stronger than the published CLRS MPNN on DFS.
- HyPARE was not retrained under the official-style no-augmentation, no-LN
  configuration.
- Only BFS and DFS are studied; no PGN-adjacency variant is trained.
- BFS is at the ceiling and cannot separate the architectures.

## References

1. P. Velickovic et al. The CLRS Algorithmic Reasoning Benchmark. ICML, 2022.
2. P. Velickovic and C. Blundell. Neural Algorithmic Reasoning. Patterns, 2021.
3. J. Gilmer et al. Neural Message Passing for Quantum Chemistry. ICML, 2017.
4. Y. Li et al. Gated Graph Sequence Neural Networks. ICLR, 2016.
5. P. Velickovic et al. Pointer Graph Networks. NeurIPS, 2020.
