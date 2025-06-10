import torch
from torch.nn import Linear, ReLU, ModuleList, Sequential, Dropout, Softmax, Tanh

class RnnModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers):
        super(RnnModel, self).__init__()
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

    def forward(self, inputs):
        hidden = torch.zeros(self.num_layers,
                             self.batch_size,
                             self.hidden_size,)
        hidden = hidden.to(self.device)
        out, _ = self.rnn(inputs, hidden)
        return out.view(1, self.hidden_size)

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out.view(1, self.hidden_size)

class MLPModel(torch.nn.Module):
    # 默认一层隐藏层
    def __init__(self, input_size, hidden_size, output_size):
        """
        :param input_size: int 输入神经元个数
        :param hidden_size,: 每层隐藏层神经元个数
        :param output_size: int 输出神经元个数
        :param num_layer: int 隐藏层层数
        :param dropout: float 训练完丢掉多少
        """
        super(MLPModel, self).__init__()
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.output_size)


