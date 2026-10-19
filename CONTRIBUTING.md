# Contributing to CrickVision

Thanks for helping improve CrickVision. The project is experimental and prioritises reproducibility, rights-aware data handling, and honest evaluation over unsupported feature claims.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dashboard]"
ruff check crickvision tests
python -m compileall -q crickvision tests
python -m pytest -q
```

On Windows, activate the environment with `.venv\\Scripts\\Activate.ps1`.

## Pull requests

- Keep changes scoped and explain the problem, design, trade-offs, and limitations.
- Add deterministic tests for bug fixes and new behavior.
- Do not commit videos, extracted frames, checkpoints, secrets, or other assets unless their redistribution rights are explicit and documented.
- Split datasets by match/source; do not tune on the held-out test split.
- Include exact commands, environment versions, and benchmark configuration for performance claims.
- Preserve the distinction between observed detections and inferred events.
- Keep unknown/ambiguous states explicit rather than forcing labels.
- Update README/docs and the model/data card when behavior or limitations change.

## Model and dataset contributions

For new weights or data, provide provenance, version/hash, applicable licence/terms, source rights, annotation schema, match-level split, and evaluation protocol. Do not submit weights trained on unlicensed footage or present a dataset card as proof that underlying broadcast footage can be redistributed.

## Reporting bugs

Include operating system, Python/PyTorch/Torchvision/OpenCV versions, exact command, redacted error output, input format/dimensions/FPS, and whether the issue reproduces with a short rights-cleared clip. Never attach private or restricted footage to a public issue.
