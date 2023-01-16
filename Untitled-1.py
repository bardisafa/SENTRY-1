class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def regenerate(self):
        self.data = torch.ones(3, 1).float()


data = torch.zeros(3, 1).float()
dataset = MyDataset(data)
print(dataset[0]) # should output 0

data_loader = DataLoader(dataset)

for d in data_loader:
    print(d)

dataset.regenerate()
print(dataset[0]) # should output 1

for d in data_loader:
    print(d)