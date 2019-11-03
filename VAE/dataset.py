from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt

class Dataset_brain(data.Dataset):
    def __init__(self,file):
        self.file=np.load(file)

    def __getitem__(self, index):
        img = self.file[index]
       # plt.imshow(img)
        #plt.show()
       # img=(img-0.5)/0.5
        return img[np.newaxis,:]

    def __len__(self):
        return len(self.file)
