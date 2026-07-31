from data.dataloader import dataloader

iterator = iter(dataloader())
inputs, targets = next(iterator)
print("Inputs:", inputs)
print("Targets:", targets)
