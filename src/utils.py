import matplotlib.pyplot as plt

def show_images(data, num_images=4):
    data_iter = iter(data)
    images, labels = next(data_iter)
    plt.figure(figsize=(8,4))
    for i in range(num_images):
        plt.subplot(1,num_images,i+1)
        plt.imshow(images[i].permute(1,2,0))
        plt.title('Cat' if labels[i] == 0 else 'Dog')
        plt.axis('off')
    plt.show()
