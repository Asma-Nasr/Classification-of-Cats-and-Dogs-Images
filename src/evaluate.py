import torch

def evaluate(model, loader,criterion, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labs)
            loss_sum += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labs).sum().item()
            total += imgs.size(0)
    return loss_sum / total, correct / total
 
