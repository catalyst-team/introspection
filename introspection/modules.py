from catalyst import utils
from catalyst.contrib.nn import ResidualBlock
from torch import nn
import torchvision


class SequentialUnwrapper(nn.Module):
    def __init__(self, nn_module):
        super().__init__()
        self.nn_module = nn_module

    def forward(self, x):
        x_ = self.nn_module(x)
        return x, x_


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def resnet9(in_channels: int, num_classes: int, size: int = 16):
    sz, sz2, sz4, sz8 = size, size * 2, size * 4, size * 8
    return nn.Sequential(
        conv_block(in_channels, sz),
        conv_block(sz, sz2, pool=True),
        ResidualBlock(nn.Sequential(conv_block(sz2, sz2), conv_block(sz2, sz2))),
        conv_block(sz2, sz4, pool=True),
        conv_block(sz4, sz8, pool=True),
        ResidualBlock(nn.Sequential(conv_block(sz8, sz8), conv_block(sz8, sz8))),
        nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Dropout(0.2),
            SequentialUnwrapper(nn.Linear(sz8, num_classes)),
        ),
    )


class TemporalResNet(nn.Module):
    def __init__(
        self,
        emb_features: int,
        out_features: int,
        dropout_p: float = 0.5,
        arch: str = "resnet18",
        pretrained: bool = True,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Flatten()
        if freeze_encoder:
            for module in resnet.children():
                utils.set_requires_grad(module, requires_grad=False)

        self.encoder = nn.Sequential(resnet, nn.Dropout(p=dropout_p))
        self.embedder = nn.Sequential(nn.Linear(in_features, emb_features), nn.ReLU())
        # self.attention = nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid())
        self.classifier = nn.Linear(emb_features, out_features)

        self.embedder.apply(utils.get_optimal_inner_init(nn.ReLU))
        self.classifier.apply(utils.outer_init)

    def forward(self, x):
        bs, ln, ch, h, w = x.shape
        x = self.encoder(x.view(-1, ch, h, w))
        embeddings = self.embedder(x.view(bs, ln, -1))
        # x_a = self.attention(x.view(bs, sl, -1))
        # x = x_r * x_a
        logits = self.classifier(embeddings).mean(1)
        return embeddings, logits
