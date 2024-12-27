import pickle
import torch
from torch.utils.data import Dataset

class MYDataset(Dataset):
    def __init__(self,d1_file,gram_data_file,label_file,train=True):
        with open(d1_file, "rb") as f:
            self.molvec= pickle.load(f)
        with open(gram_data_file, "rb") as f:
            self.gram = pickle.load(f)
        with open(label_file, "rb") as f:
            self.y_list = pickle.load(f)

    def __getitem__(self, index):
        data1=self.gram[index]
        #need  tensor type
        data1=torch.Tensor(data1)
        data1 = data1.unsqueeze(0)

        data2=self.molvec[index]
        data2 = torch.Tensor(data2)
        data2 = data2.unsqueeze(0)

        # data=torch.cat((data1,data2),-2)
        label=self.y_list[index]
        return data1,data2,label,torch.LongTensor([index])

    def __len__(self):
        return len(self.y_list)