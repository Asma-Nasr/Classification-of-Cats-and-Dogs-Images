
def train(model, loader, criterion, optimizer, device, epochs):
    model.train() 
       
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labs).sum().item()
            total += imgs.size(0)
        
        epoch_loss = running_loss / total
        epoch_accuracy = correct / total
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
