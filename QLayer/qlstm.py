import torch
import torch.nn as nn

import pennylane as qml
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from QCircuit.qcircuit import vqc


class QLSTM(nn.Module):
    def __init__(self, 
                input_size, 
                hidden_size, 
                n_qubits=4,
                n_qlayers=1,
                qembed_type="angle",
                qlayer_type="basic",
                batch_first=True,
                return_sequences=False, 
                return_state=False):
        super(QLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.qembed_type = qembed_type
        self.qlayer_type = qlayer_type

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        #weight_shapes = {"weights": (n_qlayers, n_qubits)}

        self.clayer_in = torch.nn.Linear(self.concat_size, n_qubits)
        self.VQC = [vqc(self.n_qubits, self.n_layers, self.qembed_type, self.qlayer_type) for _ in range(4)]
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)

    def forward(self, x, init_states=None):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        '''
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size)  # hidden state (output)
            c_t = torch.zeros(batch_size, self.hidden_size)  # cell state
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]
            
            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # match qubit dimension
            y_t = self.clayer_in(v_t)

            i_t = torch.sigmoid(self.clayer_out(self.VQC[0](y_t)))  # input block
            f_t = torch.sigmoid(self.clayer_out(self.VQC[1](y_t)))  # forget block
            g_t = torch.tanh(self.clayer_out(self.VQC[2](y_t)))  # update block
            o_t = torch.sigmoid(self.clayer_out(self.VQC[3](y_t))) # output block

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)


class BiQLSTM(nn.Module):
    def __init__(self, 
                input_size, 
                hidden_size, 
                n_qubits=4,
                n_qlayers=1,
                qembed_type="angle",
                qlayer_type="basic",
                batch_first=True,
                return_sequences=False, 
                return_state=False):
        super(BiQLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size // 2
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.qembed_type = qembed_type
        self.qlayer_type = qlayer_type

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.qlstm_fw = QLSTM(self.input_size, self.hidden_size, self.n_qubits, self.n_qlayers, 
                self.qembed_type, self.qlayer_type, self.batch_first, self.return_sequences, self.return_state)
        self.qlstm_bw = QLSTM(self.input_size, self.hidden_size, self.n_qubits, self.n_qlayers, 
                self.qembed_type, self.qlayer_type, self.batch_first, self.return_sequences, self.return_state)

    def forward(self, x, init_states=None):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        '''

        x_fw = x
        x_bw = x[:, ::-1, ::-1]

        if init_states is None:
            h_init = torch.zeros(batch_size, self.hidden_size)  # hidden state (output)
            c_init = torch.zeros(batch_size, self.hidden_size)  # cell state
            fw_init_states = (h_init, c_init)
            bw_init_states = (h_init, c_init)
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            fw_init_states, bw_init_states = init_states

        hidden_seq_fw, (h_t_fw, c_t_fw) = self.qlstm_fw(x_fw, fw_init_states)
        hidden_seq_bw, (h_t_bw, c_t_bw) = self.qlstm_bw(x_bw, bw_init_states)
        hidden_seq_bw = hidden_seq_bw[:, ::-1, ::-1]

        hidden_seq = torch.concat((hidden_seq_fw, hidden_seq_bw), dim=0)
        fw_state = (h_t_fw, c_t_fw)
        bw_state = (h_t_bw, c_t_bw)

        return hidden_seq, (fw_state, bw_state)

