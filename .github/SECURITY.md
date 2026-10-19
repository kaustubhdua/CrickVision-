# Security policy

CrickVision is an experimental local video-processing project.

## Supported versions

Security fixes are currently made against the default branch. There is no formal release or backport policy yet.

## Reporting a vulnerability

Please do not publish credentials, private videos, or exploit details in a public issue. Use GitHub's private vulnerability reporting feature for this repository if it is enabled; otherwise contact the repository maintainer privately through a verified contact channel.

## Current security boundaries

- The Streamlit dashboard is for one trusted local user and is not hardened for public or multi-user hosting.
- Video processing can consume substantial memory, disk, CPU, and GPU resources.
- The CLI does not provide authentication or access control; keep local media and outputs in appropriately permissioned directories.
- Do not run untrusted model checkpoints or install unreviewed dependencies.
- Review all source-video, model, and dataset terms independently.
