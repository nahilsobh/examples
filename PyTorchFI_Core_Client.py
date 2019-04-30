"""
Copyright (c) 2019 University of Illinois 
All rights reserved.

Developed by:         
                          RSIM Research Group
                          University of Illinois at Urbana-Champaign
                        http://rsim.cs.illinois.edu/
"""

import torch
from torch.autograd import Variable
import torch.nn as nn

from pytorchfi import PyTorchFI_Core

from dataset.CIFAR10_dataset import CIFAR10_CLASSES
from dataset.CIFAR10_dataset import CIFAR10_dataset
from models.AlexNet import AlexNet
from models.model_util import load_pretrained


PRETRAINED_PATH = "PATH_TO_MODEL_CHECKPOINT"
BATCH_SIZE = 1


def set_zero(self, input, output):
    output = torch.IntTensor(output.size()).zero_()


"""
EXAMPLE CLIENT FOR PYTORCHFI CORE
https://n3a9.github.io/pytorchfi-docs-beta/docs/user/core/example_client/
"""

if __name__ == "__main__":

    model = load_pretrained(AlexNet(), PRETRAINED_PATH, False)
    data = Variable(CIFAR10_dataset(BATCH_SIZE)[0])
    softmax = nn.Softmax(dim=1)

    "Golden Output"
    golden_output = model(data)
    golden_output_softmax = softmax(golden_output)
    golden = list(torch.argmax(golden_output_softmax, dim=1))

    PyTorchFI_Core.init(model, 32, 32, 1)

    "Single Specified Fault Injection"
    model = PyTorchFI_Core.declare_neuron_fi(
        conv_num=2, batch=0, c=2, h=1, w=1, value=10000000000000
    )

    single_corrupted_output = model(data)
    single_corrupted_output_softmax = softmax(single_corrupted_output)
    single_corrupted = list(torch.argmax(single_corrupted_output_softmax, dim=1))

    "List Specified Fault Injection"
    model = PyTorchFI_Core.declare_neuron_fi(
        conv_num=[1, 2],
        batch=[0, 0],
        c=[1, 2],
        h=[1, 1],
        w=[1, 1],
        value=[1392323, 10000000],
    )

    list_corrupted_output = model(data)
    list_corrupted_output_softmax = softmax(list_corrupted_output)
    list_corrupted = list(torch.argmax(list_corrupted_output_softmax, dim=1))

    "Randomized Fault Injection"
    model = PyTorchFI_Core.declare_neuron_fi()
    random_corrupted_output = model(data)
    random_corrupted_output_softmax = softmax(random_corrupted_output)
    random_corrupted = list(torch.argmax(random_corrupted_output_softmax, dim=1))

    "Custom Fault Injection"
    model = PyTorchFI_Core.declare_neuron_fi(function=set_zero)
    custom_corrupted_output = model(data)
    custom_corrupted_output_softmax = softmax(custom_corrupted_output)
    custom_corrupted = list(torch.argmax(custom_corrupted_output_softmax, dim=1))

    "Weight Injection"
    model = PyTorchFI_Core.declare_weight_fi(0, -1e-3, 1e-3)
    weight_output = model(data)
    weight_output_softmax = softmax(weight_output)
    weight_corrupted = list(torch.argmax(weight_output_softmax, dim=1))

    print("Expected vs Specified vs List Specified vs Random vs Custom vs Weight")
    for index, predicted_val in enumerate(random_corrupted):
        print(
            "%s vs %s vs %s vs %s vs %s vs %s"
            % (
                CIFAR10_CLASSES[predicted_val],
                CIFAR10_CLASSES[single_corrupted[index]],
                CIFAR10_CLASSES[list_corrupted[index]],
                CIFAR10_CLASSES[random_corrupted[index]],
                CIFAR10_CLASSES[custom_corrupted[index]],
                CIFAR10_CLASSES[weight_corrupted[index]],
            )
        )

    """Half precision model (CUDA ENABLED) """

    # model = AlexNet().cuda().half()
    # data = Variable(CIFAR10_dataset(BATCH_SIZE)[0]).cuda().half()
    # golden_output = model(data)
    # print(golden_output)
