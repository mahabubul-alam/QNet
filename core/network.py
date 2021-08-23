import torch
import torch.nn as nn
import numpy as np
import pennylane as qml
from qiskit.test.mock import FakeMelbourne
from qiskit.providers.aer.noise import NoiseModel
from qiskit import IBMQ
import qiskit.providers.aer.noise as noise

torch.manual_seed(0)

class Net(nn.Module):
    def __init__(self, n_qubits = None, n_layers = None, n_class = None, n_hidden_Q = 1, n_features = None, \
        noise = 0, shots = 1024):
        super(Net, self).__init__()
        assert n_features%n_qubits == 0
        self.n_hidden_Q = n_hidden_Q #number of hidden quantum layers
        self.n_qubits = n_qubits #number of qubits per parametric circuit
        self.n_layers = n_layers #number of layers in a parametric circuit
        self.n_class = n_class #number of classes in the dataset
        self.n_features = n_features #number of features in the dataset
      
        weight_shapes = {'weights': (self.n_layers, self.n_qubits)}

        if noise == 0:
            dev = qml.device("default.qubit", wires = self.n_qubits) #target pennylane device
            qnode = qml.QNode(self.circuit, dev, interface = 'torch', diff_method = 'adjoint') #circuit
        elif noise == 'hardware':
            QX_TOKEN = "XX"
            IBMQ.enable_account(QX_TOKEN)
            dev = qml.device('qiskit.ibmq', wires=self.n_qubits, backend='ibmq_bogota')
            qnode = qml.QNode(self.circuit, dev, interface = 'torch', diff_method = 'parameter-shift') #circuit
        else:
            noise_model = self.get_noise_model()
            dev = qml.device('qiskit.aer', wires = self.n_qubits, noise_model = noise_model, \
                shots = shots, basis_gates = noise_model.basis_gates)
            qnode = qml.QNode(self.circuit, dev, interface = 'torch', diff_method = 'parameter-shift') #circuit
        
        self.ql = nn.ModuleList([qml.qnn.TorchLayer(qnode, weight_shapes) for i in range(int(n_features/n_qubits)*n_hidden_Q)])
        self.lr = nn.ModuleList([nn.LeakyReLU(negative_slope=0.2) for i in range(int(n_features/n_qubits)*n_hidden_Q)])
        #self.drops = nn.ModuleList([nn.Dropout(p=0.33) for i in range(int(n_features/n_qubits)*n_hidden_Q)])
        self.fc1 = nn.Linear(n_features, n_class)

    def circuit(self, inputs, weights):
        assert len(inputs) == self.n_qubits
        #input data angle encoding
        for i, val in enumerate(inputs):
            qml.Hadamard(wires = i)
            qml.RZ(np.pi*val, wires = i)
        #parametric layers
        for l in range(self.n_layers):
            if self.n_qubits > 1:
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i+1)%self.n_qubits])
            for j in range(self.n_qubits):
                qml.RY(weights[l, j], wires = j)
        #qubit pauli-Z expectation values taken as outputs
        _expectations = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return _expectations

    def forward(self, X):
        for hq in range(self.n_hidden_Q):
            x = torch.tensor_split(X, int(self.n_features/self.n_qubits), dim = 1)
            segs = []
            for seg, x_segment in enumerate(x):
                x_segment = self.ql[hq * int(self.n_features/self.n_qubits) + seg](x_segment)
                segs.append(x_segment)
            X = torch.cat(segs, dim = 1)
            #shuffling the tensor
            torch.manual_seed(hq)
            X = X[:,torch.randperm(X.size()[1])]
            X = self.lr[hq](X)
            #X = self.drops[hq](X)
        X = self.fc1(X)
        return X

    @staticmethod
    def get_noise_model():
        device_backend = FakeMelbourne()
        noise_model = NoiseModel.from_backend(device_backend)
        return noise_model
    
    @staticmethod
    def get_noise_model_simple(Noise_Scaling_Factor=None):
        assert Noise_Scaling_Factor is not None
        prob_1 = 0.001*Noise_Scaling_Factor  # 1-qubit gate
        prob_2 = 0.01*Noise_Scaling_Factor   # 2-qubit gate
        error_1 = noise.depolarizing_error(prob_1, 1)
        error_2 = noise.depolarizing_error(prob_2, 2)

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1, ['rz', 'h', 'ry', 'rx'])
        noise_model.add_all_qubit_quantum_error(error_2, ['crz', 'cx', 'crx', 'cz'])

        return noise_model
