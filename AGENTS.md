# AGENTS.md

## Quick Start

```bash
pip install -r requirements.txt
python data/fineweb.py --max-shards 20    # download ~2B tokens
python train.py --run_id ATTNRAW-V1-L4     # run an experiment
```

## Key Commands

- `python train.py --run_id <RUN_ID> [--max_tokens N]` — single-run entry point
- `bash launch_distributed.sh` — auto-detects GPUs, runs experiment matrix sequentially with resume safety
- `bash launch_distributed.sh --fresh` — start fresh (delete state)
- `bash launch_local.sh` — sequential single-GPU launcher for all experiments
- `CUDA_VISIBLE_DEVICES=0 python train.py ... & CUDA_VISIBLE_DEVICES=1 python train.py ... &` — manual 2-GPU

## Run IDs

All run IDs are defined in `configs/runs.py`. Available optimizers: `muon`, `simpleavg_v1`, `attnraw_v1`, `attnraw_v1_g`, `attnraw_v2`, `attnraw_v3`.

## Architecture

- `train.py` — platform-agnostic entry point; uses `torch.compile`
- `configs/runs.py` — experiment matrix (RUNS dict, TRAIN_CONFIG, MODEL_CONFIG)
- `optimizers/` — custom optimizers (all follow `torch.optim.Optimizer`)
- `model/gpt.py` — nanoGPT ~44M (6 layers, 8 heads, 512 dim)
- `data/fineweb.py` — dataloader for HuggingFace FineWeb

Muon and gradient-history optimizers apply only to non-embedding params; embeddings always use Adam.

## Outputs

- Checkpoints: `checkpoints/<run_id>/ckpt_latest.pt`, `ckpt_final.pt`
- Logs: `logs/<run_id>/metrics.jsonl`

## Notes

- `analyze_results.py` in the README does not exist
- No lint/typecheck/test infrastructure — experimental research code
- Training uses `FINEWEB_MAX_SHARDS` env var (default 20 shards if unset)
- `launch_distributed.sh` coordinates via `experiment_state.json` with file locking — safe to interrupt and resume
