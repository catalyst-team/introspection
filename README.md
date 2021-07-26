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
