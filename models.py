import torch
import torch.nn as nn
import math
import torch.nn.functional as F


#  Model 1 Linear
# =======================
class LinearModel(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_d)
            #
        )

    def forward(self, x):
        return self.net(x)


# Model 2  +4 скрытых слоя
# =======================
class MLPModel(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout):
        super().__init__()

        layers = [
            nn.Flatten(),
            nn.Linear(seq_len * input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        ]

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        layers.append(nn.Linear(hidden_size, out_d))
        # layers.append()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


#  Модель 3, GRU
# ==================================================
class GRUModel(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.post_rnn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size // 2)
        )
        self.output = nn.Linear(hidden_size // 2, out_d)


    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        last_out = rnn_out[:, -1, :]
        norm_out = self.layer_norm(last_out)
        processed = self.post_rnn(norm_out)
        out = self.output(processed)
        return out


#  Model 4 модель, 1d+lstm(128, 3cлоя) со старой тренировкой
# =======================
class EnhancedRNNModel(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Linear(64, out_d)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        last_out = attn_out[:, -1, :]
        return self.output(last_out)



# Model 5 TransformerModel
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout, nhead=8, dim_feedforward=256):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Linear(64, out_d)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.pooling(x).squeeze(2)
        return self.fc(x)



# Model 6 EnhancedRNNModelV2 Conv1d
class EnhancedRNNModelV2(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Linear(64, out_d)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


# Model 7 TCNModel
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout, kernel_size=5):
        super().__init__()
        num_channels = [64 * (2 ** i) for i in range(num_layers)]
        layers = []
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                                        dilation=dilation_size,
                                        padding=(kernel_size - 1) * dilation_size,
                                        dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], out_d)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)
        out = out[:, :, -1]
        out = self.fc(out)  # Пропускаем через выходной слой
        return out


# Model 8 — NBeatsModel
class NBeatsBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.backcast = nn.Linear(hidden_size, input_size)
        self.forecast = nn.Linear(hidden_size, 4)  # это не используется в основном NBeatsModel

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.backcast(x), self.forecast(x)


class NBeatsModel(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout, nb_blocks=4):
        super().__init__()
        self.input_size = seq_len * input_size
        self.hidden_size = hidden_size

        self.input_fc = nn.Linear(self.input_size, hidden_size)
        self.blocks = nn.ModuleList([self._make_block(hidden_size, num_layers) for _ in range(nb_blocks)])

        self.final_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Linear(64, out_d),

        )

    def _make_block(self, hidden_size, nb_layers):
        layers = [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        for _ in range(nb_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)  # (B, 180 * 6)
        x = self.input_fc(x)  # (B, hidden)
        for block in self.blocks:
            x = block(x)
        return self.final_fc(x)


# Model 9 — SimpleInformer
class SimpleInformer(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout, n_heads=8, d_layers=3,
                 d_ff=256):
        super().__init__()
        self.out_len = out_d

        self.input_proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, n_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        decoder_layer = nn.TransformerDecoderLayer(hidden_size, n_heads, d_ff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, d_layers)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Linear(64, out_d),

        )

    def forward(self, x_enc, x_dec=None):
        enc_out = self.input_proj(x_enc)  # (B, seq_len, d_model)
        enc_out = self.encoder(enc_out)
        if x_dec is None:
            x_dec = torch.zeros((x_enc.size(0), self.out_len, x_enc.size(2)), device=x_enc.device)
        dec_out = self.input_proj(x_dec)
        dec_out = self.decoder(dec_out, enc_out)
        return self.projection(dec_out[:, -1, :])  # (B, 3)


# Model 9 TFTLite
class TFTLite(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout, num_heads=2,
                 num_lstm_layers=1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_lstm_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)

        # Строим FC-слои динамически
        fc_layers = []
        for _ in range(num_layers - 1):
            fc_layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
        fc_layers.append(nn.Linear(hidden_size, out_d))

        self.output = nn.Sequential(*fc_layers)


    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = attn_out[:, -1, :]
        out = self.output(out)
        return out

# Model 10 SimpleDeepAR
class SimpleDeepAR(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Linear(64, out_d),

        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


# Model 11 PatchTST
class PatchTST(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout, patch_len=5, nhead=4):
        super().__init__()
        self.patch_len = patch_len
        self.embed_dim = hidden_size

        self.patch_proj = nn.Linear(input_size * patch_len, hidden_size)
        self.positional_encoding = nn.Parameter(torch.zeros(500, hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Linear(64, out_d),

        )

    def forward(self, x):
        B, T, C = x.shape
        num_patches = T // self.patch_len
        x = x[:, :num_patches * self.patch_len, :]
        x = x.reshape(B, num_patches, self.patch_len * C)

        x = self.patch_proj(x) + self.positional_encoding[:x.size(1)]
        x = self.transformer_encoder(x)

        x = x.mean(dim=1)
        return self.output_layer(x)


# Model 12 TimesNetClassifier
class TimesBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class TimesNetClassifier(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout):
        super().__init__()
        self.input_proj = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.blocks = nn.Sequential(*[TimesBlock(hidden_size, hidden_size) for _ in range(num_layers)])
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Linear(64, out_d),

        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.pooling(x)
        return self.fc(x)


# Model 13 GRUD
class GRUD(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_d),

        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


# Model 14 SCINet
class Interactor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class SCINet(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.interactor = Interactor(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_d),

        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        x = self.interactor(x)
        x = x.mean(dim=2)
        return self.fc(x)


# Model 15 FEDformer
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc_real = nn.Linear(in_channels, out_channels)
        self.fc_imag = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x_ft = torch.fft.rfft(x, dim=1)
        x_ft_real = x_ft.real
        x_ft_imag = x_ft.imag
        x = self.fc_real(x_ft_real) + self.fc_imag(x_ft_imag)
        return x


class FEDformer(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList([
            FourierBlock(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_d),

        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        return self.fc(x)


# Model 16 EnhancedRNNModelNew
class EnhancedRNNModelNew(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, out_d),

        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        last_out = attn_out[:, -1, :] + lstm_out[:, -1, :]
        return self.output(last_out)


# Model 17 TransformerModelNew
class PositionalEncodingNew(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModelNew(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout, nhead=8, dim_feedforward=256):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncodingNew(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, out_d),

        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.pooling(x).squeeze(2)
        return self.fc(x)


# Model 18 EnhancedRNNModelV2New
class EnhancedRNNModelV2New(nn.Module):
    def __init__(self, seq_len, input_size,  out_d, num_layers, hidden_size, dropout):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(256, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, out_d),

        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

