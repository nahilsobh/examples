"""
Copyright (c) 2019 University of Illinois
All rights reserved.

Developed by:         
                          RSIM Research Group
                          University of Illinois at Urbana-Champaign
                        http://rsim.cs.illinois.edu/
"""

from dataset.CIFAR10_dataset import CIFAR10_CLASSES
from dataset.CIFAR10_dataset import CIFAR10_dataset
from models.AlexNet import AlexNet

from pytorchfi import PyTorchFI_Util

PRETRAINED_PATH = "PATH_TO_MODEL_CHECKPOINT"
BATCH_SIZE = 1

"""
EXAMPLE CLIENT FOR PYTORCHFI UTIL
https://n3a9.github.io/pytorchfi-docs-beta/docs/user/util/example_client/
"""

if __name__ == "__main__":
    PyTorchFI_Util.init(AlexNet(), CIFAR10_dataset(BATCH_SIZE))
    PyTorchFI_Util.declare_neuron_fi(
        conv_num=2, batch=0, c=2, h=1, w=1, value=10000000000000
    )

    results = PyTorchFI_Util.compare_golden()
    print("Expected vs Corrupted")
    for index, predicted_val in enumerate(results[0]):
        print(
            "%s vs. %s"
            % (CIFAR10_CLASSES[predicted_val], CIFAR10_CLASSES[results[1][index]])
        )

    x = PyTorchFI_Util.random_batch_fi_gen(2, 2, 1, 1, 1000, 1000000)
    print(x)
