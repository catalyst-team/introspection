import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor):
        return self.block(x) + x


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float = 0.5,
        hidden_size: int = 128,
        num_layers: int = 0,
    ):
        super(MLP, self).__init__()
        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(p=dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers.append(
                ResidualBlock(
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                    )
                )
            )
        layers.append(
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, output_size),
            )
        )

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        bs, ln, fs = x.shape
        fc_output = self.fc(x.view(-1, fs))
        fc_output = fc_output.view(bs, ln, -1).mean(1)  # .squeeze(1)
        return fc_output


class LSTM(nn.Module):
    def __init__(
        self,
        output_size: int,
        fc_dropout: float = 0.5,
        hidden_size: int = 128,
        bidirectional: bool = False,
        **kwargs,
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            hidden_size=hidden_size, bidirectional=bidirectional, **kwargs
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=fc_dropout),
            nn.Linear(2 * hidden_size if bidirectional else hidden_size, output_size),
        )

    def forward(self, x):
        lstm_output, _ = self.lstm(x)

        if self.bidirectional:
            out_forward = lstm_output[:, -1, : self.hidden_size]
            out_reverse = lstm_output[:, 0, self.hidden_size :]
            lstm_output = torch.cat((out_forward, out_reverse), 1)
        else:
            lstm_output = lstm_output[:, -1, :]

        fc_output = self.fc(lstm_output)
        return fc_output


class Transformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        fc_dropout: float = 0.5,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_heads: int = 8,
    ):
        super(Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            transformer_encoder,
        ]
        self.transformer = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Dropout(p=fc_dropout), nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        fc_output = self.transformer(x)
        fc_output = fc_output[:, -1, :]
        # fc_output = fc_output.mean(1)
        fc_output = self.fc(fc_output)
        return fc_output
