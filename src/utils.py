import matplotlib.pyplot as plt

def show_imaages(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        # Convert from (C, H, W) to (H, W, C)
        plt.imshow(images[i].permute(1, 2, 0))  
        plt.title('Cat' if labels[i] == 0 else 'Dog')  
        plt.axis('off')
    plt.show()
