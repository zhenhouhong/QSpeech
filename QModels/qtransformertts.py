import torch
import torch.nn as nn

import pennylane as qml
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
#sys.path.append(os.path.abspath(os.path.join(__dir__, '../'..)))

from QLayer.qtransformer import QTransformerBlock
from QLayer.qconv import QCONV_5x5, QCONV_1x1


class EncoderPreNet(nn.Module):
    def __init__(self, embed_dims, hidden_dims, dropout):
        super(EncoderPreNet, self).__init__()
        self.conv1 = nn.Conv1d(
            embed_dims,
            hidden_dims,
            kernel_size=5, 
            padding=2
        )

        self.conv2 = nn.Conv1d(
            hidden_dims,
            hidden_dims,
            kernel_size=5, 
            padding=2
        )

        self.conv3 = nn.Conv1d(
            hidden_dims,
            hidden_dims,
            kernel_size=5, 
            padding=2
        )

        self.batch_norm1 = nn.BatchNorm1d(hidden_dims)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dims)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dims)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hidden_dims, hidden_dims)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout1(torch.relu(self.batch_norm1(self.conv1(x))))
        x = self.dropout2(torch.relu(self.batch_norm2(self.conv2(x))))
        x = self.dropout3(torch.relu(self.batch_norm3(self.conv3(x))))
        x = x.transpose(1, 2)
        x = self.fc_out(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dims,
        hidden_dims,
        max_len,
        heads,
        forward_expansion,
        num_layers, 
        dropout,
        n_qubits,
        n_qlayers,
        qembed_type,
        qlayer_type
    ):
        super(Encoder, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dims)
        self.positional_embed = nn.Parameter(torch.zeros(1, max_len, hidden_dims))
        self.prenet = EncoderPreNet(embed_dims, hidden_dims, dropout)
        self.dropout = nn.Dropout(dropout)
        self.attention_layers = nn.Sequential(
            *[
                QTransformerBlock(
                    hidden_dims, 
                    heads, 
                    dropout, 
                    forward_expansion,
                    n_qubits=n_qubits,
                    n_qlayers=n_qlayers,
                    qembed_type=qembed_type,
                    qlayer_type=qlayer_type
                )
                for _ in range(num_layers)
            ]
        )
    
    def forward(self, x, mask=None):
        seq_len = x.shape[1]
        token_embed = self.token_embed(x)
        positional_embed = self.positional_embed[:, :seq_len, :]
        x = self.prenet(token_embed)
        x += positional_embed
        x = self.dropout(x)
        for layer in self.attention_layers:
            x = layer(x, x, x, mask)
        return x


class DecoderPreNet(nn.Module):
    def __init__(self, mel_dims, hidden_dims, dropout):
        super(DecoderPreNet, self).__init__()
        self.fc_out = nn.Sequential(
            nn.Linear(mel_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):

        return self.fc_out(x)


class PostNet(nn.Module):
    def __init__(self, mel_dims, hidden_dims, dropout):
        #causal padding -> padding = (kernel_size - 1) x dilation
        #kernel_size = 5 -> padding = 4
        #Exclude the last padding_size output as we want only left padded output
        super(PostNet, self).__init__()
        self.conv1 = nn.Conv1d(mel_dims, hidden_dims, kernel_size=5, padding=4)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dims)
        self.dropout1 = nn.Dropout(dropout)
        self.conv_list = nn.Sequential(
            *[
                nn.Conv1d(hidden_dims, hidden_dims, kernel_size=5, padding=4)
                for _ in range(3)
            ]
        )

        self.batch_norm_list = nn.Sequential(
            *[
                nn.BatchNorm1d(hidden_dims)
                for _ in range(3)
            ]
        )
        
        self.dropout_list = nn.Sequential(
            *[
                nn.Dropout(dropout)
                for _ in range(3)
            ]
        )

        self.conv5 = nn.Conv1d(hidden_dims, mel_dims, kernel_size=5, padding=4)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout1(torch.tanh(self.batch_norm1(self.conv1(x)[:, :, :-4])))
        for dropout, batchnorm, conv in zip(self.dropout_list, self.batch_norm_list, self.conv_list):
            x = dropout(torch.tanh(batchnorm(conv(x)[:, :, :-4])))
        out = self.conv5(x)[:, :, :-4]
        out = out.transpose(1, 2)
        return out


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dims,
        heads,
        forward_expansion,
        dropout,
        n_qubits=4,
        n_qlayers=1,
        qembed_type="angle",
        qlayer_type="basic"
    ):
        super(DecoderBlock, self).__init__()
        self.causal_masked_attention = QSELFATTENTION(
            embed_dims, 
            heads,
            n_qubits=n_qubits,
            n_qlayers=n_qlayers,
            qembed_type=qembed_type,
            qlayer_type=qlayer_type)
        self.attention_layer = QTransformerBlock(
            embed_dims, 
            heads, 
            dropout, 
            forward_expansion,
            n_qlayers=n_qlayers,
            n_qubits=n_qubits,
            qembed_type=qembed_type,
            qlayer_type=qlayer_type
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dims)
    
    def forward(self, query, key, value, src_mask, causal_mask):
        causal_masked_attention = self.causal_masked_attention(query, query, query, causal_mask)
        query = self.dropout(self.layer_norm(causal_masked_attention + query))
        out = self.attention_layer(query, key, value, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        mel_dims,
        hidden_dims,
        heads,
        max_len,
        num_layers,
        forward_expansion,
        dropout,
        n_qubits=4,
        n_qlayers=1,
        qembed_type="angle",
        qlayer_type="basic"
    ):
        super(Decoder, self).__init__()
        self.positional_embed = nn.Parameter(torch.zeros(1, max_len, hidden_dims))
        self.prenet = DecoderPreNet(mel_dims, hidden_dims, dropout)
        self.attention_layers = nn.Sequential(
            *[
                DecoderBlock(
                    hidden_dims, 
                    heads, 
                    forward_expansion, 
                    dropout,
                    n_qubits=n_qubits,
                    n_qlayers=n_qlayers,
                    qembed_type=qembed_type,
                    qlayer_type=qlayer_type
                )
                for _  in range(num_layers)
            ]
        )
        self.mel_linear = nn.Linear(hidden_dims, mel_dims)
        self.stop_linear = nn.Linear(hidden_dims, 1)
        self.postnet = PostNet(mel_dims, hidden_dims, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, mel, encoder_output, src_mask, casual_mask):
        seq_len = mel.shape[1]
        prenet_out = self.prenet(mel)
        x = self.dropout(prenet_out + self.positional_embed[:, :seq_len, :])

        for layer in self.attention_layers:
            x = layer(x, encoder_output, encoder_output, src_mask, casual_mask)

        stop_linear = self.stop_linear(x)

        mel_linear = self.mel_linear(x)

        postnet = self.postnet(mel_linear)

        out = postnet + mel_linear

        return out, mel_linear, stop_linear


class QTransformerTTS(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dims,
        hidden_dims, 
        heads,
        forward_expansion,
        num_layers,
        dropout,
        mel_dims,
        max_len,
        pad_idx,
        n_qubits=4,
        n_qlayers=1,
        qembed_type="angle",
        qlayer_type="basic"
    ):
        super(QTransformerTTS, self).__init__()
        self.encoder = Encoder(
            vocab_size,
            embed_dims,
            hidden_dims,
            max_len,
            heads,
            forward_expansion,
            num_layers,
            dropout,
            n_qubits=n_qubits,
            n_qlayers=n_qlayers,
            qembed_type=qembed_type,
            qlayer_type=qlayer_type
        )
        
        self.decoder = Decoder(
            mel_dims,
            hidden_dims,
            heads,
            max_len,
            num_layers,
            forward_expansion,
            dropout,
            n_qubits=n_qubits,
            n_qlayers=n_qlayers,
            qembed_type=qembed_type,
            qlayer_type=qlayer_type
        )

        self.pad_idx = pad_idx

    def target_mask(self, mel, mel_mask):
        seq_len = mel.shape[1]
        pad_mask = (mel_mask != self.pad_idx).unsqueeze(1).unsqueeze(3)
        causal_mask = torch.tril(torch.ones((1, seq_len, seq_len))).unsqueeze(1)
        return pad_mask, causal_mask
    
    def input_mask(self, x):
        mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def forward(self, text_idx, mel, mel_mask):
        input_pad_mask = self.input_mask(text_idx)
        target_pad_mask, causal_mask = self.target_mask(mel, mel_mask)
        encoder_out = self.encoder(text_idx, input_pad_mask)
        mel_postout, mel_linear, stop_linear = self.decoder(mel, encoder_out, target_pad_mask, causal_mask)
        return mel_postout, mel_linear, stop_linear


if __name__ == "__main__":
    a = torch.randint(0, 30, (4, 60))
    mel = torch.randn(4, 128, 80)
    mask = torch.ones((4, 128))
    model = QTransformerTTS(
        vocab_size=30,
        embed_dims=512,
        hidden_dims=256, 
        heads=4,
        forward_expansion=4,
        num_layers=6,
        dropout=0.1,
        mel_dims=80,
        max_len=512,
        pad_idx=0
    )
    x, y, z = model(a, mel, mask)
    print(x.shape, y.shape, z.shape)
