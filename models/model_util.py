import os
import torch
from collections import OrderedDict


def load_pretrained(model, path, use_cuda):
    assert os.path.isfile(path), "Error: invalid directory given! %s" % path
    state_dict = OrderedDict()

    if use_cuda:
        checkpoint = torch.load(path)
        for k, v in checkpoint["state_dict"].items():
            name = "module." + k.replace("module.", "")
            state_dict[name] = v
    else:
        checkpoint = torch.load(path, map_location="cpu")
        for k, v in checkpoint["state_dict"].items():
            name = k.replace("module.", "")
            state_dict[name] = v

    model.load_state_dict(state_dict)
    return model
