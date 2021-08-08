import os
import sys
from torch.utils.data import Dataset
import numpy as np
import torch
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


class MyCSVDatasetReader(Dataset):

    def __init__(self, csv_path, PC = 0):
        
        if not os.path.isfile(csv_path):
            print('Please provide a valid path of the dataset')
            return

        self.DATA = np.genfromtxt(csv_path,delimiter=',')
        self.X = self.DATA[:,0:-1]
        if int(PC) > 0:
            pca_model = PCA(n_components = int(PC), random_state = 0)
            pca_model.fit(self.X)
            self.X = pca_model.transform(self.X)


        self.Y = self.DATA[:,-1]
        self.X = normalize(self.X, axis = 0, norm = 'max')
        print(self.X)
        

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        X = torch.FloatTensor(self.X[idx,:])
        Y = self.Y[idx]
        sample = {'feature': X, 'label': Y}
        return sample


    def get_labels(self):
        return self.Y
    
    def get_number_of_class(self):
        return len(np.unique(self.Y)) 

    def get_number_of_features(self):
        return self.X.shape[1]


if __name__ == '__main__':

    assert len(sys.argv) == 3
    dataset = MyCSVDatasetReader(str(sys.argv[1]))
    sample = dataset.__getitem__(int(sys.argv[2]))
    print(sample)
    print(dataset.get_number_of_class())
    print(dataset.get_number_of_features())


