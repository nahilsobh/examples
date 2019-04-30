import torch
import torchvision
import torchvision.transforms as transforms

"""
Derived from https://discuss.pytorch.org/t/what-is-the-recommended-way-to-preprocess-cifar10-and-100/20846
"""

TRANSFORM_TEST = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

CIFAR10_CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def CIFAR10_dataset(batch_size):
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=TRANSFORM_TEST
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=1
    )
    data = iter(test_loader)
    images, labels = data.next()
    return [images, batch_size]
