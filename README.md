# Introspection [WIP]

### tl;dr

#### Prerequirements
```bash
pip install -r ./requirements.txt
```

#### Image
```bash
# classification
PYTHONPATH=. python ./image/train.py
# classification + metric learning
PYTHONPATH=. python ./image/train.py --use-ml
```

#### Video
```bash
# go through data/scripts/readme.md
# classification
# ~11GB GPU required ;)
PYTHONPATH=. python ./video/train.py
# classification + metric learning
# ~20GB GPU required ;)
PYTHONPATH=. python ./video/train.py --use-ml
```

#### Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RE-592Tp4xjAOaxrMOI8EdmjzWEOxPOI?usp=sharing) Captum visualization