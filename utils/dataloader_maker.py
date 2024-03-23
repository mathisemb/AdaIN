from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torchvision.transforms as transforms

class ArtBench10(CIFAR10):
    base_folder = "artbench-10-batches-py"
    url = "https://artbench.eecs.berkeley.edu/files/artbench-10-python.tar.gz"
    filename = "artbench-10-python.tar.gz"
    tgz_md5 = "9df1e998ee026aae36ec60ca7b44960e"
    train_list = [
        ["data_batch_1", "c2e02a78dcea81fe6fead5f1540e542f"],
        ["data_batch_2", "1102a4dcf41d4dd63e20c10691193448"],
        ["data_batch_3", "177fc43579af15ecc80eb506953ec26f"],
        ["data_batch_4", "566b2a02ccfbafa026fbb2bcec856ff6"],
        ["data_batch_5", "faa6a572469542010a1c8a2a9a7bf436"],
    ]
    test_list = [
        ["test_batch", "fa44530c8b8158467e00899609c19e52"],
    ]
    meta = {
        "filename": "meta",
        "key": "styles",
        "md5": "5bdcafa7398aa6b75d569baaec5cd4aa",
    }

def dataloader_maker(nb_of_images: int = 512, batch_size: int = 64):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    # datasets
    cifar10_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
    artbench_dataset = ArtBench10(root='./data', train=True, transform=transform, download=True)

    # choose the size of the training dataset
    indices = list(range(nb_of_images))
    cifar10_dataset = Subset(cifar10_dataset, indices)
    artbench_dataset = Subset(artbench_dataset, indices)

    # create the dataloaders
    cifar10_loader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=True)
    artbench_loader = DataLoader(artbench_dataset, batch_size=batch_size, shuffle=True)

    #print("len(cifar10_dataset) =", len(cifar10_dataset), "images")
    #print("len(artbench_dataset) =", len(artbench_dataset), "images")

    #print("len(cifar10_loader) =", len(cifar10_loader), "batches")
    #print("len(artbench_loader) =", len(artbench_loader), "batches")

    return cifar10_loader, artbench_loader