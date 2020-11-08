import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from customDataset import CharlieDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_channel = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 32
num_epochs = 1

# data loading
dataset = CharlieDataset(csv_file = 'customDataset.csv', root_dir = 'charlies',
                         transform = transforms.ToTensor())
print('taille : ', len(dataset))
train_set, test_set = torch.utils.data.random_split(dataset, [2, 2])
train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True)

# model
model = torchvision.models.googlenet(pretrained=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#training

for epoch in range(num_epochs):
    losses=[]
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        scores = model(data)
        loss = criterion(scores, targets)
        
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
    #print(f'Cost at epoch {epoch} is {sum(losses).len(Losses)}')
    
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}')
        
    model.train()
    
print("checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on test Set")
check_accuracy(test_loader, model)
        
    
        

