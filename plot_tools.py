import matplotlib.pyplot as plt
import numpy as np

def plot_img(content, style, stylized):
    plt.figure(figsize=(15, 5))

    # Display the content image
    plt.subplot(1, 3, 1)
    content_np = content.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    plt.imshow(content_np)
    plt.title('Content Image')
    plt.axis('off')

    # Display the style image
    plt.subplot(1, 3, 2)
    style_np = style.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    plt.imshow(style_np)
    plt.title('Style Image')
    plt.axis('off')

    # Display the stylized image
    plt.subplot(1, 3, 3)
    stylized_np = stylized.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    plt.imshow(stylized_np)
    plt.title('Stylized Image')
    plt.axis('off')

    plt.show()

