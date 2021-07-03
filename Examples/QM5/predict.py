import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

import matplotlib.pyplot as plt
import IPython.display as ipd

import test_config as test_cfg
from dataloader import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchaudio.datasets import SPEECHCOMMANDS
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__dir__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__), "../.."))

from QModels import QM5
from utils import utils

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
waveform_first, *_ = train_set[0]
waveform_second, *_ = train_set[1]
waveform_last, *_ = train_set[-1]

#Formatting the Data
new_sample_rate = cfg.sample_rate
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)

batch_size = test_cfg

if "cuda" == device:
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

# Define the Network
model = QM5(n_input=transformed.shape[0], n_output=len(labels))
model.to(device)

# Load checkpoint if exists
try:
    checkpoint = torch.load(os.path.join(test_cfg.checkpoint_path,'{}_{}.pth.tar'.format(test_cfg.model_name, test_cfg.restore_step)))
    model.load_state_dict(checkpoint['model'])
    print("\n--------model load at step %d--------\n" % cfg.restore_step)
except FileNotFoundError as fnf_error:
    print("\n--------Must load a model--------\n")

# The transform needs to live on the same device as the model and the data.
transform = transform.to(device)


def predict(tensor):
    # Use the model to predict the label of the waveform
    tensor = tensor.to(device)
    tensor = transform(tensor)
    tensor = model(tensor.unsqueeze(0))
    tensor = get_likely_index(tensor)
    tensor = index_to_label(tensor.squeeze())
    return tensor


waveform, sample_rate, utterance, *_ = train_set[-1]
ipd.Audio(waveform.numpy(), rate=sample_rate)

print(f"Expected: {utterance}. Predicted: {predict(waveform)}.")


for i, (waveform, sample_rate, utterance, *_) in enumerate(test_set):
    output = predict(waveform)
    if output != utterance:
        ipd.Audio(waveform.numpy(), rate=sample_rate)
        print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")
        break
else:
    print("All examples in this dataset were correctly classified!")
    print("In this case, let's just look at the last data point")
    ipd.Audio(waveform.numpy(), rate=sample_rate)
    print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")

