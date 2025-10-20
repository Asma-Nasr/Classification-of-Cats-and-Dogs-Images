from torchvision import datasets,transforms
from torch.utils.data import DataLoader

def load_data():
 
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
 
    train_data = datasets.ImageFolder(root='train',transform=transform)
    test_data = datasets.ImageFolder(root='test',transform=transform)
 
    train_loader = DataLoader(train_data,batch_size=16,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=16,shuffle=False)
    
    return train_data,test_data,train_loader,test_loader
