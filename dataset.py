import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MovingMNIST(Dataset):
    def __init__(self, dataset_path, n_frames=16, norm_mean=0, norm_std=1):
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.data = torch.from_numpy(np.float32(np.load(dataset_path)))
        #Dataset will be of the form (L, T, C, H, W)
        self.data = self.data.permute(1, 0, 2, 3).unsqueeze(2)
        self.n_frames = n_frames #This can't be greater than 20
        
        #self.normalize = transforms.Normalize(self.norm_mean, self.norm_std)
        self.normalize = lambda x: (x - 128)/128
        self.denormalize = lambda x: x*128 + 128

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, i):
        T = self.data.size(1)
        ot = np.random.randint(T - self.n_frames) if T > self.n_frames else 0
        x = self.data[i, ot:(ot + self.n_frames)]
        x = self.normalize(x)
        return x


if __name__ == "__main__":
    dset = MovingMNIST("mnist_test_seq.npy")
    zero = dset[0]
    print(type(zero))
    print(zero.size())
