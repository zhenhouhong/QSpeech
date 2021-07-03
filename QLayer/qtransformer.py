import torch
import torch.nn as nn

import pennylane as qml

from QLayer.qselfattention import QSelfAttention


class QTransformerBlock(nn.Module):

    def __init__(self, 
                 hidden_dims, 
                 heads, 
                 dropout, 
                 forward_expansion,
                 n_qubits=4,
                 n_qlayers=1,
                 qembed_type="angle",
                 qlayer_type="basic"):
        super(TransformerBlock, self).__init__()
        self.hidden_dims = hidden_dims
        self.heads = heads
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.qembed_type = qembed_type
        self.qlayer_type = qlayer_type
        self.multihead_attention = QSELFATTENTION(hidden_dims,
                                                  heads,
                                                  n_qubits=self.n_qubits,
                                                  n_qlayers=self.n_qlayers,
                                                  qembed_type=self.qembed_type,
                                                  qlayer_type=self.qlayer_type)
        self.feed_forward = nn.Sequential(
            nn.Conv1d(hidden_dims, hidden_dims*forward_expansion, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dims*forward_expansion, hidden_dims, kernel_size=1)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dims)
        self.layer_norm2 = nn.LayerNorm(hidden_dims)
    
    def forward(self, query, key, value, mask):
        attention_out = self.multihead_attention(query, key, value, mask)
        add = self.dropout(self.layer_norm1(attention_out + query))
        ffn_in = add.transpose(1, 2)
        ffn_out = self.feed_forward(ffn_in)
        ffn_out = ffn_out.transpose(1, 2)
        out = self.dropout(self.layer_norm2(ffn_out + add)) 
        return out
