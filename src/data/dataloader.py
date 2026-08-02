from torch.utils.data import DataLoader
from data.dataset import Dataset

def create_dataloader():
    # Example usage of the Dataset class
    txt = "This is an example text for the dataset."
    context_length = 2
    stride = 2

    dataset = Dataset(txt, context_length, stride)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    return loader
