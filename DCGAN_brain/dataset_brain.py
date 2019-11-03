from torch.utils import data
import numpy as np
class Dataset_brain(data.Dataset):
    def __init__(self,file):
        self.file=np.load(file)
        # file_seg=file.replace('t2','seg')
        # self.label=np.load(file_seg)
    def __getitem__(self, index):
        img=self.file[index]
        img=(img-0.5)/0.5
        # label=self.label[index]
        return img[np.newaxis,:] #,label[np.newaxis,:]
    def __len__(self):
        return len(self.file)