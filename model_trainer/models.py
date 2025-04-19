
import torch
import torch.nn as nn
import copy

class TitleRegression(nn.Module):
    def __init__(self, input_size=1024, output_size = 3, layer_neurons = [2048, 1024, 512], dropout = 0.3):
        super(TitleRegression, self).__init__()

        self.shared_layers = nn.Sequential()
        prev_size = input_size
        
        for i, neurons in enumerate(layer_neurons[:-1]): 
            self.shared_layers.add_module(f"linear_{i}", nn.Linear(prev_size, neurons))
            self.shared_layers.add_module(f"bn_{i}", nn.BatchNorm1d(neurons))
            self.shared_layers.add_module(f"leakyrelu_{i}", nn.LeakyReLU())
            self.shared_layers.add_module(f"dropout_{i}", nn.Dropout(dropout))
            prev_size = neurons

        self.regression_head = nn.Sequential(
            nn.Linear(prev_size, layer_neurons[-1]),
            nn.BatchNorm1d(layer_neurons[-1]),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_neurons[-1], output_size)
        )

        self.reconstruction_head = nn.Sequential(
            nn.Linear(prev_size, layer_neurons[-1]),
            nn.BatchNorm1d(layer_neurons[-1]),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_neurons[-1], input_size) 
        )

    def forward(self, x, task_type = 'regression'):
        shared_out = self.shared_layers(x)
        
        if task_type == 'reconstruction':
            out = self.reconstruction_head(shared_out)
        else:
            regression_out = self.regression_head(shared_out)
            out = torch.sigmoid(regression_out) * 100.0  
        return out

class SentenceRegression(nn.Module):
    def __init__(self, vec_num = 6, input_size = 1024, output_size = 2, layer_neurons = [2048, 1024, 512], dropout = 0.3):
        super(SentenceRegression, self).__init__()

        self_network = nn.Sequential()
        prev_size = input_size
        for i, neurons in enumerate(layer_neurons[:-1]): 
            self_network.add_module(f"linear_{i}", nn.Linear(prev_size, neurons))
            self_network.add_module(f"bn_{i}", nn.BatchNorm1d(neurons))
            self_network.add_module(f"leakyrelu_{i}", nn.LeakyReLU())
            self_network.add_module(f"dropout_{i}", nn.Dropout(dropout))
            prev_size = neurons

        reconstruction_head = nn.Sequential(
            nn.Linear(prev_size, layer_neurons[-1]),
            nn.BatchNorm1d(layer_neurons[-1]),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_neurons[-1], input_size) 
        )

        self.vec_num = vec_num
        self.input_size = input_size
        self.output_size = output_size
        self.first_reconstruction_size = prev_size
        self.self_network = nn.ModuleList([copy.deepcopy(self_network) for _ in range(vec_num)])
        self.reconstruction_head = nn.ModuleList([copy.deepcopy(reconstruction_head) for _ in range(vec_num)])
        self.regression_head = nn.Sequential(
            nn.Linear(prev_size * vec_num, layer_neurons[-1]),
            nn.BatchNorm1d(layer_neurons[-1]),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_neurons[-1], output_size)
        )

    def forward(self, x, task_type = 'regression'):
        # x shape: (batch_size, vec_num * input_size)
        feature_outputs = [ net(x[:, i*self.input_size:(i+1)*self.input_size]) for i, net in enumerate(self.self_network)]
        feature_outputs = torch.stack(feature_outputs, dim = 0) 
        feature_outputs = torch.permute(feature_outputs, (1, 0, 2)).reshape(x.shape[0], -1)

        if task_type == 'reconstruction':
            # out shape: (batch_size, vec_num * input_size)
            reconstruction_outputs = [ net(feature_outputs[:, i*self.first_reconstruction_size:(i+1)*self.first_reconstruction_size]) for i, net in enumerate(self.reconstruction_head)]
            reconstruction_outputs = torch.stack(reconstruction_outputs, dim = 0) 
            out = torch.permute(reconstruction_outputs, (1, 0, 2)).reshape(x.shape[0], -1)
        else:
            # out shape: (batch_size, output_size)
            regression_out = self.regression_head(feature_outputs)
            out = torch.stack([torch.tanh(regression_out[:, 0]), torch.sigmoid(regression_out[:, 1])], dim = 1) * 100.0
        return out


class FakeNewsModel(nn.Module):

    def __init__(self, bert_input_size: int, other_input_size: int,output_size: int = 2, layer_neurons = [1024, 512, 256], dropout: int = 0.25):
        super(FakeNewsModel, self).__init__()

        self.inner_layers = nn.Sequential()
        prev_size = other_input_size + bert_input_size
        
        for i, neurons in enumerate(layer_neurons[:-1]): 
            self.inner_layers.add_module(f"linear_{i}", nn.Linear(prev_size, neurons))
            self.inner_layers.add_module(f"bn_{i}", nn.BatchNorm1d(neurons))
            self.inner_layers.add_module(f"leakyrelu_{i}", nn.LeakyReLU())
            self.inner_layers.add_module(f"dropout_{i}", nn.Dropout(dropout))
            prev_size = neurons

        self.classification_layer = nn.Sequential(
            nn.Linear(prev_size, layer_neurons[-1]),
            nn.BatchNorm1d(layer_neurons[-1]),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_neurons[-1], output_size)
        )

    def forward(self, x):
        out = self.inner_layers(x)
        out = self.classification_layer(out)
        return out
