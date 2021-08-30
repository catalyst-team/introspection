# Introspection [WIP]

### tl;dr

#### Prerequirements
```bash
pip install -r ./requirements.txt
```

#### Image
```bash
# classification
PYTHONPATH=. python ./scripts/train_image.py

# classification + metric learning
PYTHONPATH=. python ./scripts/train_image.py --use-ml
```

#### Video
```bash
# go through data/scripts/readme.md
# classification
PYTHONPATH=. python ./scripts/train_video.py  # ~11GB GPU required
PYTHONPATH=. python ./scripts/train_video.py --freeze-encoder  # ~5GB GPU required

# classification + metric learning
PYTHONPATH=. python ./scripts/train_video.py --use-ml  # ~20GB GPU required
PYTHONPATH=. python ./scripts/train_video.py --freeze-encoder --use-ml  # ~20GB GPU required
```

#### Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RE-592Tp4xjAOaxrMOI8EdmjzWEOxPOI?usp=sharing) Captum visualization