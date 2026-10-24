# Online resource discovery

CrickVision includes `crickvision-research`, a small public-metadata scout for discovering potentially relevant cricket computer-vision resources without API keys.

```bash
crickvision-research
crickvision-research --limit 20 --json > runs/resource-scout.json
```

The command queries public metadata APIs for:
- Hugging Face datasets matching `cricket`
- Hugging Face models matching `cricket`
- GitHub repositories matching cricket computer vision and tracking

It prints names, links, API-exposed licence metadata, update/download indicators, and per-source availability. If one service is unavailable, it still reports results from the others. Use `--limit` to request 1–30 results per source and `--timeout` to change the per-source network timeout.

## Important limits

- The scout does **not** download, install, or execute third-party code, weights, datasets, or videos.
- An API licence tag is only a discovery hint. Open the resource card/repository and audit the exact terms, model provenance, labels, data splits, and underlying broadcast-footage rights before using it.
- A dataset of still images cannot on its own validate persistent player identity, ball trajectories, or temporal events.
- Results depend on public API availability and can change over time. No API key is required.
- The command does not silently change CrickVision's detector or claim a discovered model is compatible. Integrations must go through a separate licence, dependency, and evaluation review.

This tool makes research reproducible and reduces manual discovery work; it does not replace cricket-specific evaluation on held-out, labelled match data.
