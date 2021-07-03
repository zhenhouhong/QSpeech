import torch


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

