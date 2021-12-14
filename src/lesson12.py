import torch
from torchsummary import summary

# batch_size = 1
# seq_len = 3
# input_size = 4
# hidden_size = 2
#
# cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
#
# dataset = torch.randn(seq_len, batch_size, input_size)
# hidden = torch.zeros(batch_size, hidden_size)
#
# for idx, inputs in enumerate(dataset):
#     print('=' * 20, idx, '=' * 20)
#     print("InputSize: ", inputs.shape)
#     hidden = cell(inputs, hidden)
#     print('OutputSize: ', hidden.shape)
#     print(hidden)

class Mode(torch.nn.Module):
    def __init__(self, inputSize, hiddenSize, batchSize, embeddingSize, numLayers, numClass):
        super(Mode, self).__init__()
        self.embeddingSize = embeddingSize
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.batchSize = batchSize
        self.numLayers = numLayers
        self.numClass = numClass
        self.emb = torch.nn.Embedding(self.inputSize, self.embeddingSize)
        self.rnn = torch.nn.RNN(input_size=self.inputSize,
                                hidden_size=self.hiddenSize,
                                num_layers=self.numLayers,
                                batch_first=True)
        self.fc = torch.nn.Linear(self.hiddenSize, numClass)

    def forward(self, x):
        hidden = torch.zeros(self.numLayers, x.size(0), self.hiddenSize)
        x = self.emb(x)
        x, _ = self.rnn(x, hidden)
        x = self.fc(x)
        return x.view(-1, self.numClass)

net = Mode(4, 8, 1, 10, 2, 4)

summary(net, (1, 5))





#
#
# batch_size = 1
# input_size = 4
# hidden_size = 2
#
#

