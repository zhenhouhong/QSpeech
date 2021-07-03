import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

import config as cfg
from dataloader import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchaudio.datasets import SPEECHCOMMANDS
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, "../..")))

from QModels.qm5 import QM5
from utils import utils

#print(len(train_set))

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
waveform_first, *_ = train_set[0]
waveform_second, *_ = train_set[1]
waveform_last, *_ = train_set[-1]

#Formatting the Data
new_sample_rate = cfg.sample_rate
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)

batch_size = cfg.batch_size

if "cuda" == device:
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10

# Make checkpoint directory if not exists
if not os.path.exists(cfg.checkpoint_path):
    os.mkdir(cfg.checkpoint_path)

# Load checkpoint if exists
try:
    checkpoint = torch.load(os.path.join(cfg.checkpoint_path,'checkpoint_%d.pth.tar'% args.restore_step))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("\n--------model restored at step %d--------\n" % cfg.restore_step)
except:
    print("\n--------Start New Training--------\n")

current_step = 0


def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        current_step = batch_idx + cfg.restore_step

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        # accuracy
        pred = utils.get_likely_index(output)
        correct += utils.number_of_correct(pred, target)
        pred_acc = 100.0 * correct / data.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            print("Train Epoch: {epoch}, Accuracy: {pred_acc:.6f}")

        # record loss
        losses.append(loss.item())

        # save checkpoint
        if current_step % cfg.save_step == 0:
            utils.save_checkpoint({'model':model.state_dict(),
                'optimizer':optimizer.state_dict()},
                os.path.join(cfg.checkpoint_path,'{}_{}.pth.tar'.format(cfg.model_name, current_step)))
            print("save model at step %d ..." % current_step)


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)
        loss = F.nll_loss(output.squeeze(), target)

        pred = utils.get_likely_index(output)
        correct += utils.number_of_correct(pred, target)

    print(f"\nTest Epoch: {epoch}\tLoss: {loss.item():.6f}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")


log_interval = cfg.log_interval
n_epoch = cfg.epochs

losses = []

# The transform needs to live on the same device as the model and the data.
transform = transform.to(device)
for epoch in range(1, n_epoch + 1):
    train(model, epoch, log_interval)
    #train(model, epoch, log_interval, current_step)
    test(model, epoch)
    scheduler.step()

