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
PYTHONPATH=. python ./scripts/train_video.py --freeze-encoder --use-ml  # ~15GB GPU required
```

#### Time Series [WIP]
```
# unpack https://drive.google.com/file/d/1MKtGfcVNiq0p7T_XGYGyTYFDYiUH96fS/view?usp=sharing to data/abide

PYTHONPATH=. python ./scripts/train_ts.py
PYTHONPATH=. python ./scripts/train_ts.py --use-ml
```

#### Demo

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)] (https://colab.research.google.com/drive/1Uo2lTQJCo31ZAejl1y7tyt8BGRE6wcxj#scrollTo=CDa4TjwSou33) [Training pipeline](./notebooks/train.ipynb)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/introspection/blob/main/notebooks/intospection.ipynb) [Captum visualization](./notebooks/intospection.ipynb)
