from torch.utils.data import DataLoader
from data.dataset import Dataset

def create_dataloader(txt, batch_size, max_context_length, stride, drop_last, shuffle, num_workers):
    # Example usage of the Dataset class
    context_length = max_context_length
    stride = stride

    dataset = Dataset(txt, context_length, stride)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, 
        num_workers=num_workers
    )

    return loader
