import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def _cycle_loader(loader):
    """Creates an infinite iterator from a DataLoader."""
    while True:
        for data in loader:
            yield data


def get_cifar10_iterator(batch_size, data_root='./sample_graph'):
    """
    Gets an infinite-looping iterator for the CIFAR-10 dataset.

    Args:
        batch_size (int): The batch size.
        data_root (str): The root directory for data storage.

    Returns:
        iterator: An iterator that returns tuples of (images, labels).
                  Images are torch.Tensors with shape (N, 3, 32, 32) and pixel values in [-1, 1].
    """
    # Define the data preprocessing pipeline.
    # 1. Convert PIL Image or numpy.ndarray to torch.FloatTensor of shape (C, H, W) with values in [0.0, 1.0].
    # 2. Normalize pixel values from [0.0, 1.0] to [-1.0, 1.0].
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load the training dataset.
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                            download=True, transform=transform)

    # Create the PyTorch DataLoader.
    # - shuffle=True: Shuffles the data at the beginning of each epoch.
    # - num_workers: Use multiple subprocesses to load data, speeding up the process.
    # - drop_last=True: Drops the last incomplete batch if its size is less than batch_size.
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=2, drop_last=True)

    # Return an infinite-looping iterator.
    return _cycle_loader(trainloader)


# --- Test Code ---
# The following code will execute when you run `python data_loader.py` directly.
# This is very useful for quickly testing if the data loading works correctly.
if __name__ == '__main__':
    print("Testing the data loader...")

    # Set a small batch size for the test.
    test_batch_size = 4

    # Get the data iterator.
    cifar10_iterator = get_cifar10_iterator(batch_size=test_batch_size)

    # Get one batch of data from the iterator.
    images, labels = next(cifar10_iterator)

    # Print output to verify the data format.
    print("Successfully fetched a batch of data!")
    print(f"Type of the image batch (images): {type(images)}")
    print(f"Shape of the image batch: {images.shape}")  # Should be (4, 3, 32, 32)
    print(f"Type of the label batch (labels): {type(labels)}")
    print(f"Shape of the label batch: {labels.shape}")  # Should be (4,)
    print(f"Minimum pixel value: {images.min():.2f}")  # Should be close to -1.0
    print(f"Maximum pixel value: {images.max():.2f}")  # Should be close to 1.0