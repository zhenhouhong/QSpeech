import torch
import torch.nn as nn

import pennylane as qml
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from QCircuit.qcircuit import vqc


class QSelfAttention(nn.Module):
    def __init__(self, 
                 embed_dims,
                 heads,
                 kdim=None,
                 vdim=None,
                 n_qubits=4,
                 n_qlayers=1,
                 qembed_type="angle",
                 qlayer_type="basic"):
        super(QSelfAttention, self).__init__()
        self.heads = heads
        self.embed_dims = embed_dims
        self.depth = embed_dims//heads
        self.kdim = kdim if kdim is not None else self.depth
        self.vdim = vdim if vdim is not None else self.depth
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.qembed_type = qembed_type
        self.qlayer_type = qlayer_type

        self.VQC = [vqc(self.n_qubits, self.n_qlayers, self.qembed_type, self.qlayer_type) for _ in range(3)]
        
        self.qlayer_in = torch.nn.Linear(self.depth, self.n_qubits)
        self.klayer_in = torch.nn.Linear(self.kdim, self.n_qubits)
        self.vlayer_in = torch.nn.Linear(self.vdim, self.n_qubits)
        
        self.query = nn.Linear(self.n_qubits, self.depth)
        self.key = nn.Linear(self.n_qubits, self.kdim)
        self.value = nn.Linear(self.n_qubits, self.vdim)

        self.fc_out = nn.Linear(self.depth*self.heads*2, self.embed_dims)
    
    def forward(self, query, key, value, mask):
        batch, q_len, k_len, v_len = query.shape[0], query.shape[1], key.shape[1], value.shape[1]

        query = query.reshape(batch, q_len, self.heads, self.depth)
        key = key.reshape(batch, k_len, self.heads, self.depth)
        value = value.reshape(batch, v_len, self.heads, self.depth)
        #query = query.reshape(self.heads, self.depth, batch, q_len)
        #key = key.reshape(self.heads, self.depth, batch, k_len)
        #value = value.reshape(self.heads, self.depth, batch, v_len)

        query_qlist = []
        for i in range(q_len):
            query_hlist = []
            for j in range(self.heads):
                query_in = self.qlayer_in(query[:, i, j, :])
                query_q = self.VQC[0](query_in)
                query_hlist.append(torch.unsqueeze(query_q, 1))
            query_hcc = torch.cat(query_hlist, 1)
            query_qlist.append(torch.unsqueeze(query_hcc, 1))
        query_qcc = torch.cat(query_qlist, 1)

        key_klist = []
        for i in range(k_len):
            key_hlist = []
            for j in range(self.heads):
                key_in = self.klayer_in(key[:, i, j, :])
                key_q = self.VQC[1](key_in)
                key_hlist.append(torch.unsqueeze(key_q, 1))
            key_hcc = torch.cat(key_hlist, 1)
            key_klist.append(torch.unsqueeze(key_hcc, 1))
        key_qcc = torch.cat(key_klist, 1)

        value_vlist = []
        for i in range(v_len):
            value_hlist = []
            for j in range(self.heads):
                value_in = self.vlayer_in(value[:, i, j, :])
                value_q = self.VQC[2](value_in)
                value_hlist.append(torch.unsqueeze(value_q, 1))
            value_hcc = torch.cat(value_hlist, 1)
            value_vlist.append(torch.unsqueeze(value_hcc, 1))
        value_qcc = torch.cat(value_vlist, 1)

        query = self.query(query_qcc)
        key = self.key(key_qcc)
        value = self.value(value_qcc)

        energy = torch.einsum('bqhd, bkhd -> bhqk', [query, key])
        
        if mask is not None:
            energy.masked_fill(mask==0, float("-1e20"))

        energy =  torch.softmax((energy/((self.depth**1/2))), dim=-1)

        out = torch.einsum('bhqv, bvhd -> bqhd', [energy, value])

        out = out.reshape(batch, q_len, self.heads*self.depth)
        query = query.reshape(batch, q_len, self.heads*self.depth)

        out = torch.cat([query, out], dim=-1)
        out = self.fc_out(out)

        return out
