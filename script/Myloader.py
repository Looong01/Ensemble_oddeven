from torchvision import transforms,datasets
from torch.utils.data import DataLoader, Dataset,TensorDataset



def data_loader(image_dir,  batch_size, drop_last=False, shuffle=True):
    data = datasets.ImageFolder(image_dir, transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()]))
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle ,drop_last=drop_last)
    num_data = len(data)
    return loader, num_data