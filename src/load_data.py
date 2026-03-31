import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
 
def get_dataloaders(data_dir, batch_size = 16, train_ratio = 0.8, seed = 42):
  
 # create transfroms
 train_transform = transforms.Compose([
  transforms.Resize((224,224)),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor()
 ])
 test_transform = transforms.Compose([
  transforms.Resize((224,224)),
  transforms.ToTensor()
 ]) 
 
 # loading the full dataset
 full_dataset = datasets.ImageFolder(data_dir)

 # reproducible split
 torch.manual_seed(seed)
 
 # creating the splits
 train_split = int(train_ratio * len(full_dataset))
 test_split = len(full_dataset) - train_split

 train_data, test_data = random_split(full_dataset, [train_split, test_split])
 
 #applying transforms
 train_data.dataset.transform = train_transform
 test_data.dataset.transform = test_transform
 
 # creating data loaders
 train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
 test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

 return train_loader, test_loader


 
