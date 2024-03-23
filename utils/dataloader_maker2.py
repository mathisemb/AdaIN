from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
import torchvision.transforms as transforms
from torchvision import transforms, datasets

def dataloader_maker(folder_path ='data/MS_COCO_val', nb_of_images=4096, batch_size=8): # The nb_of_images is 80k in the paper
    transform = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=folder_path, transform=transform)

    # Create a sampler to select a subset of nb_of_images from the dataset
    sampler = RandomSampler(dataset, num_samples=nb_of_images, replacement=True)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return dataloader