import torch


import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import (GINConv,
                                global_add_pool)

class GINNet(torch.nn.Module):
    def __init__(self):
        super(GINNet, self).__init__()
        x_num_features = 37
        e_num_features = 11
        dim = 108
        self.dim = dim
        self.embedding1 = torch.nn.Embedding(10, 5)
        self.embedding2 = torch.nn.Embedding(4, 2)
        self.embedding3 = torch.nn.Embedding(12, 6)
        self.embedding4 = torch.nn.Embedding(7, 3)
        self.embedding5 = torch.nn.Embedding(10, 5)
        self.embedding6 = torch.nn.Embedding(9, 4)
        self.embedding7 = torch.nn.Embedding(12, 6)
        self.embedding8 = torch.nn.Embedding(2, 1)
        self.embedding9 = torch.nn.Embedding(2, 1)

        self.edge_embedding1 = torch.nn.Embedding(7, 3)
        self.edge_embedding2 = torch.nn.Embedding(5, 2)
        self.edge_embedding3 = torch.nn.Embedding(6, 3)
        self.edge_embedding4 = torch.nn.Embedding(2, 1)
        self.edge_embedding5 = torch.nn.Embedding(2, 1)

        x_nn1 = Sequential(Linear(x_num_features, dim), ReLU(), Linear(dim, dim))
        self.x_conv1 = GINConv(x_nn1)
        self.x_bn1 = torch.nn.BatchNorm1d(dim)
        x_nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.x_conv2 = GINConv(x_nn2)
        self.x_bn2 = torch.nn.BatchNorm1d(dim)

        self.e_linear = Sequential(Linear(e_num_features, 22),
                                 ReLU(),
                                 Linear(22, 40),
                                 ReLU(),
                                 Linear(40, 40))

        self.linear = Sequential(Linear(256, 512),
                                 ReLU(),
                                 Linear(512, 512),
                                 ReLU(),
                                 Linear(512, 256),
                                 ReLU(),
                                 Linear(256, 1))

    def forward(self, data):
        x, edge_index, edge_attr, edge_batch = data.x, data.edge_index, data.edge_attr, data.edge_batch

        x1 = self.embedding1(x[:, 0].long())
        x2 = self.embedding2(x[:, 1].long())
        x3 = self.embedding3(x[:, 2].long())
        x4 = self.embedding4(x[:, 3].long())
        x5 = self.embedding5(x[:, 4].long())
        x6 = self.embedding6(x[:, 5].long())
        x7 = self.embedding7(x[:, 6].long())
        x8 = self.embedding8(x[:, 7].long())
        x9 = self.embedding9(x[:, 8].long())
        x10 = x[:, 9:-3]
        x11 = data.pos/10
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11), 1)

        ee1 = edge_attr[:, 0].long()
        edge_attr1 = self.edge_embedding1(edge_attr[:, 0].long())
        edge_attr2 = self.edge_embedding2(edge_attr[:, 1].long())
        edge_attr3 = self.edge_embedding3(edge_attr[:, 2].long())
        edge_attr4 = self.edge_embedding4(edge_attr[:, 3].long())
        edge_attr5 = self.edge_embedding5(edge_attr[:, 4].long())
        edge_attr6 = edge_attr[:, 5:] / 10
        edge_attr = torch.cat((edge_attr1,
                               edge_attr2,
                               edge_attr3,
                               edge_attr4,
                               edge_attr5,
                               edge_attr6), 1)

        x = F.relu(self.x_conv1(x, edge_index))
        x = self.x_bn1(x)
        x = F.relu(self.x_conv2(x, edge_index))
        x = self.x_bn2(x)

        e = self.e_linear(edge_attr)
        x = x[edge_index].transpose(0, 1).reshape(-1, self.dim * 2)
        xe = torch.cat((x, e), 1)
        xe = self.linear(xe)
        out = global_add_pool(xe, edge_batch)

        return out


