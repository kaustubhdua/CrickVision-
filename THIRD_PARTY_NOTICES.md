# Third-party notices

The original CrickVision source code and project-authored documentation are licensed under MIT; see the root LICENSE.

Runtime dependencies currently declared in pyproject.toml include:
- PyTorch: https://github.com/pytorch/pytorch
- Torchvision: https://github.com/pytorch/vision
- OpenCV headless Python package: https://github.com/opencv/opencv-python
- Pillow: https://github.com/python-pillow/Pillow
- NumPy: https://github.com/numpy/numpy
- Streamlit (optional dashboard): https://github.com/streamlit/streamlit

Development dependencies include pytest, Ruff, and mypy. Exact package versions and transitive dependency licences should be inventoried for a release; this file is not a complete software bill of materials.

The default detector uses Torchvision Faster R-CNN ResNet-50 FPN COCO-pretrained weights. Model code, checkpoint distribution, underlying training data, and input video can have different terms. Review the exact upstream model/weights documentation and applicable notices before redistribution or commercial use.

CrickVision does not bundle broadcast footage or a labelled cricket benchmark. Candidate external datasets must be audited subset by subset, including rights in underlying footage. A public dataset card or repository licence does not necessarily grant rights to redistribute embedded third-party media.

Third-party names and links are for provenance and review; inclusion here is not an endorsement. Keep this file updated when dependencies, models, datasets, or assets change.
