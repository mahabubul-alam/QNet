from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import os

from . load_data import MyCSVDatasetReader as CSVDataset
from . network import Net

torch.manual_seed(0)

class QNet:
    def __init__(self, datafilepath = None, n_layers = None, \
        n_hidden_Q = None, lr = 0.5, bs = 30, n_epochs = 50, \
            n_qubits = 2, noise = 0, shots = 1024, repeat = 1, \
                PCA = 0):

        assert os.path.isfile(datafilepath)
        assert noise is float or int and (noise >= 0)

        self.dataset = CSVDataset(datafilepath, PCA)
        if int(PCA) > 0:
            self.out_prefix = datafilepath.split('/')[-1].split('.')[0] + f'_{PCA}PC'
        else:
            self.out_prefix = datafilepath.split('/')[-1].split('.')[0]
        self.out_prefix = 'results' + '/' + self.out_prefix

        self.out_repeat_id = None
        self.n_class = self.dataset.get_number_of_class()

        self.n_features = self.dataset.get_number_of_features()
        self.n_qubits = n_qubits
        assert self.n_features%self.n_qubits == 0

        self.n_layers = n_layers
        self.n_hidden_Q = n_hidden_Q
        self.outdir = (
            (
                (
                    f'{self.out_prefix}/{self.out_prefix}_{str(self.n_hidden_Q)}'
                    + 'H_'
                )
                + str(int(self.n_qubits))
                + 'Q_'
            )
            + str(int(self.n_layers))
            + 'L'
        )

        os.makedirs(self.outdir, exist_ok = True)

        self.noise = noise
        self.shots = shots
        self.n_epochs = n_epochs
        self.lr = lr
        self.bs = bs
        self.repeat = repeat
        self.set_model()
        self.split_data()

    def set_model(self):
        self.Network = Net(
            n_qubits = self.n_qubits,
            n_layers = self.n_layers,
            n_class = self.n_class,
            n_hidden_Q = self.n_hidden_Q,
            n_features = self.n_features,
            noise = self.noise,
            shots = self.shots,
            )

        self.device = 'cpu' 
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Network = self.Network.to(self.device)
        self.loss_fun = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adagrad(self.Network.parameters(), lr = self.lr)
    
    def reset_model_params(self):
        torch.manual_seed(self.out_repeat_id)
        for layer in self.Network.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        torch.manual_seed(torch.initial_seed())

    def split_data(self):
        kfold = StratifiedKFold(n_splits = 2, shuffle = True, random_state = 0)
        kfolds = kfold.split(list(range(self.dataset.__len__())), self.dataset.get_labels())
        splits = next(kfolds)
        self.train_ids = splits[0]
        self.test_ids = splits[1]

    def train_and_test_network(self):
        for repeat_id in range(self.repeat):
            self.out_repeat_id = repeat_id
            self.reset_model_params()
            self.train()
            self.test_network()
    
    def train_network(self):
        for repeat_id in range(self.repeat):
            self.out_repeat_id = repeat_id
            self.reset_model_params()
            self.train()

    def train(self):
        train_set = Subset(self.dataset, self.train_ids)
        train_loader = DataLoader(train_set, batch_size = self.bs, shuffle = True)
        tr_losses = []
        tr_accs = []

        test_set = Subset(self.dataset, self.test_ids)
        test_loader = DataLoader(test_set, batch_size = self.bs, shuffle = True)
        test_losses = []
        test_accs = []

        for epoch in range(self.n_epochs):
            
            t1 = time.time()
            self.Network.train()
            tr_loss = 0
            y_trues = []
            y_preds = []

            for i, sampled_batch in enumerate(train_loader):
                data = sampled_batch['feature'].type(torch.FloatTensor)
                y = sampled_batch['label'].squeeze().type(torch.LongTensor)
                data = data.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                output = self.Network(data)
                loss = self.loss_fun(output,y)
                loss.backward()
                self.optimizer.step()

                tr_loss = tr_loss + loss.data.cpu().numpy()
                y_trues += y.cpu().numpy().tolist()
                y_preds += output.data.cpu().numpy().argmax(axis = 1).tolist()

            accs = accuracy_score(y_trues, y_preds)
            tr_accs.append(accs)
            cnf = confusion_matrix(y_trues, y_preds)
            tr_loss = tr_loss/(i + 1)
            tr_losses.append(tr_loss)

            print(cnf)
            print(f'Epoch: {epoch} TR_Loss: {tr_loss} TR_Accs: {accs}')
            print(f'Time per training epoch: {time.time() - t1}')

            self.Network.eval()
            test_loss = 0
            y_trues = []
            y_preds = []

            for i, sampled_batch in enumerate(test_loader):
                data = sampled_batch['feature'].type(torch.FloatTensor)
                y = sampled_batch['label'].squeeze().type(torch.LongTensor)
                data = data.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    output = self.Network(data)
                    loss = self.loss_fun(output,y)

                test_loss = test_loss + loss.data.cpu().numpy()
                y_trues += y.cpu().numpy().tolist()
                y_preds += output.data.cpu().numpy().argmax(axis = 1).tolist()

            accs = accuracy_score(y_trues, y_preds)
            test_accs.append(accs)
            cnf = confusion_matrix(y_trues, y_preds)
            test_loss = test_loss/(i + 1)
            test_losses.append(test_loss)

            print(cnf)
            print(f'Epoch: {epoch} TEST_Loss: {test_loss} TEST_Accs: {accs}')



        np.save(
            f'{self.outdir}/'
            + f'trainset_cost_{self.noise}'
            + f'_{str(self.out_repeat_id)}'
            + '.npy',
            np.array(tr_losses),
        )

        np.save(
            f'{self.outdir}/'
            + f'trainset_accs_{self.noise}'
            + f'_{str(self.out_repeat_id)}'
            + '.npy',
            np.array(tr_accs),
        )


        torch.save(
            self.Network.state_dict(),
            f'{self.outdir}/train_model' + f'_{str(self.out_repeat_id)}',
        )


        np.save(
            f'{self.outdir}/'
            + f'testset_cost_{self.noise}'
            + f'_{str(self.out_repeat_id)}'
            + '.npy',
            np.array(test_losses),
        )

        np.save(
            f'{self.outdir}/'
            + f'testset_accs_{self.noise}'
            + f'_{str(self.out_repeat_id)}'
            + '.npy',
            np.array(test_accs),
        )

        
    def test_network(self):
        self.test(ids = self.test_ids)

    def test_trained_network(self):
        for repeat_id in range(self.repeat):
            self.out_repeat_id = repeat_id
            try:
                self.Network.load_state_dict(
                    torch.load(
                        (
                            (f'{self.outdir}/' + 'train_model')
                            + f'_{str(self.out_repeat_id)}'
                        )
                    )
                )

            except:
                return
            self.inference()

    def test(self, ids = None):
        assert ids is not None
        test_set = Subset(self.dataset, ids)
        test_loader = DataLoader(test_set, batch_size = self.bs, shuffle = True)

        self.Network.eval()
        test_loss = 0
        y_trues = []
        y_preds = []

        for i, sampled_batch in enumerate(test_loader):
            data = sampled_batch['feature'].type(torch.FloatTensor)
            y = sampled_batch['label'].squeeze().type(torch.LongTensor)
            data = data.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                output = self.Network(data)
                loss = self.loss_fun(output, y)               

            test_loss = test_loss + loss.data.cpu().numpy()
            y_trues += y.cpu().numpy().tolist()
            y_preds += output.data.cpu().numpy().argmax(axis = 1).tolist()

        accs = accuracy_score(y_trues, y_preds)
        cnf = confusion_matrix(y_trues, y_preds)
        test_loss = test_loss/(i + 1)

        print(cnf)
        print(f'Test_Loss: {test_loss} Test_Accs: {accs}')

        if list(ids) == list(self.train_ids):
            np.save(
                (
                    (
                        f'{self.outdir}/'
                        + f'inference_under_noise_{self.noise}_trainset_cost'
                        + f'_{str(self.out_repeat_id)}'
                    )
                    + '.npy'
                ),
                np.array(test_loss),
            )

            np.save(
                (
                    (
                        f'{self.outdir}/'
                        + f'inference_under_noise_{self.noise}_trainset_accs'
                        + f'_{str(self.out_repeat_id)}'
                    )
                    + '.npy'
                ),
                np.array(accs),
            )

        elif list(ids) == list(self.test_ids):
            np.save(
                (
                    (
                        f'{self.outdir}/'
                        + f'inference_under_noise_{self.noise}_testset_cost'
                        + f'_{str(self.out_repeat_id)}'
                    )
                    + '.npy'
                ),
                np.array(test_loss),
            )

            np.save(
                (
                    (
                        f'{self.outdir}/'
                        + f'inference_under_noise_{self.noise}_testset_accs'
                        + f'_{str(self.out_repeat_id)}'
                    )
                    + '.npy'
                ),
                np.array(accs),
            )

    
    def inference(self):
        print('Starting inference!')
        self.test(ids = self.train_ids)
        self.test(ids = self.test_ids)

    


if __name__ == '__main__':
    #import pdb
    #pdb.set_trace()
    for q in [4]:
        for hid in [1, 2, 3, 4]:
            for layer in [1, 2, 3, 4]:
                QNetwork = QNet(datafilepath = 'digits.csv', n_layers = layer, n_hidden_Q = hid, \
                    lr = 0.5, bs = 30, n_epochs = 20, noise = 0, shots = 1024, repeat = 1, n_qubits = q)
                #print([i for i in QNetwork.Network.parameters()])
                #cProfile.run('QNetwork.train_network()')
                QNetwork.train_network()
                #print([i for i in QNetwork.Network.parameters()])
