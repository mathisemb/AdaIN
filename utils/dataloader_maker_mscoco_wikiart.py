from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
import torchvision.transforms as transforms
from torchvision import transforms, datasets

# Define the mean RGB value for mean subtraction (From VGGTraining)
mean_rgb = [0.485, 0.456, 0.406]

def dataloader_maker(folder_path ='data/MS_COCO_val', nb_of_images=4096, batch_size=8): # The nb_of_images is 80k in the paper
    
    # Add mean subtraction and multiplication by 255 to the transformations
    transform = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_rgb, std=[1, 1, 1]),  # Subtract mean
        lambda x: x * 255  # Multiply by 255
    ])

    dataset = datasets.ImageFolder(root=folder_path, transform=transform)

    # Create a sampler to select a subset of nb_of_images from the dataset
    sampler = RandomSampler(dataset, num_samples=nb_of_images, replacement=True)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return dataloader


mean_rgb = [0.485, 0.456, 0.406]
def undo_normalization(tensor):
    # First, divide by 255 to bring the values back to the range [0, 1]
    tensor = tensor/255.0

    # Undo mean subtraction
    for i in range(3):  # Iterate over RGB channels
        tensor[:, i, :, :] += mean_rgb[i]

    # Clamp the tensor to ensure it stays within valid range [0, 1]
    #tensor = TF.normalize(tensor, mean=[0, 0, 0], std=[1/255, 1/255, 1/255])

    return tensor
