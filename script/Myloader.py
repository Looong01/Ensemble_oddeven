from torchvision import transforms,datasets
from torch.utils.data import DataLoader, Dataset,TensorDataset
from utils import *

def data_loader(target_transform, image_dir, batch_size, drop_last=False, shuffle=True):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    data = datasets.ImageFolder(image_dir, transform=transform, target_transform=target_to_oh)
    # 创建数据加载器
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    num_data = len(data)
    
    return loader, num_data