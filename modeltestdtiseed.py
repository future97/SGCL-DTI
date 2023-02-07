# -*- coding: utf-8 -*-
from utilsdtiseed import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from GCNLayer import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size).apply(init),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False).apply(init)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)


class HANLayer(nn.Module):

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):

        super(HANLayer, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GraphConv(in_size, out_size, activation=F.relu).apply(init))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                    g, meta_path)
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[0](new_g, h).flatten(1))

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, dropout, num_heads=1):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.predict = nn.Linear(hidden_size * num_heads, out_size, bias=False).apply(init)
        self.layers.append(
            HANLayer(meta_paths, in_size, hidden_size, num_heads, dropout)
        )

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
        return self.predict(h)


class HAN_DTI(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout):
        super(HAN_DTI, self).__init__()
        self.sum_layers = nn.ModuleList()

        for i in range(0, len(all_meta_paths)):
            self.sum_layers.append(
                HAN(all_meta_paths[i], in_size[i], hidden_size[i], out_size[i], dropout))
        # self.dtrans = nn.Sequential(nn.Linear(out_size, 100), nn.ReLU())
        # self.ptrans = nn.Sequential(nn.Linear(out_size, 400), nn.ReLU())

    def forward(self, s_g, s_h_1, s_h_2):
        h1 = self.sum_layers[0](s_g[0], s_h_1)
        h2 = self.sum_layers[1](s_g[1], s_h_2)
        return h1, h2


class GCN(nn.Module):
    def __init__(self, nfeat, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, 256)
        self.gc2 = GraphConvolution(256, 128)
        self.dropout = dropout

    def forward(self, x, adj):
        x = x.to(device)
        adj = adj.to(device)
        x1 = F.relu(self.gc1(x, adj), inplace=True)
        x1 = F.dropout(x1, self.dropout)
        x2 = self.gc2(x1, adj)
        res = x2
        return res


class CL_GCN(nn.Module):
    def __init__(self, nfeat, dropout,alpha = 0.8):
        super(CL_GCN, self).__init__()
        self.gcn1 = GCN(nfeat, dropout)
        self.gcn2 = GCN(nfeat, dropout)
        self.tau = 0.5
        self.alpha = alpha

    def forward(self, x1, adj1, x2, adj2, clm):
        z1 = self.gcn1(x1, adj1)
        z2 = self.gcn2(x2, adj2)

        loss = self.alpha*self.sim(z1, z2, clm) + (1-self.alpha)*self.sim(z2,z1,clm)
        return z1, z2, loss

    def sim(self, z1, z2, clm):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)

        sim_matrix = sim_matrix / (torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-8)
        sim_matrix = sim_matrix.to(device)

        loss = -torch.log(sim_matrix.mul(clm).sum(dim=-1)).mean()
        return loss

    def mix2(self, z1, z2):
        loss = ((z1 - z2) ** 2).sum() / z1.shape[0]
        return loss

class MLP(nn.Module):
    def __init__(self, nfeat):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(nfeat, 32, bias=False).apply(init),
            nn.ELU(),
            nn.Linear(32, 2, bias=False),
            nn.LogSoftmax(dim=1)
            # nn.Sigmoid())
    def forward(self, x):
        output = self.MLP(x)
        return output


class HMTCL(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout):
        super(HMTCL, self).__init__()
        self.HAN_DTI = HAN_DTI(all_meta_paths, in_size, hidden_size, out_size, dropout)
        self.CL_GCN = CL_GCN(256, dropout)
        self.MLP = MLP(256)

    def forward(self, graph, h, cl, dateset_index, data, iftrain=True, d=None, p=None):
        if iftrain:
            d, p= self.HAN_DTI(graph, h[0], h[1])
        edge, feature = constructure_graph(data, d, p)
        f_edge, f_feature = constructure_knngraph(data, d, p)
        feature1, feature2, cl_loss1 = self.CL_GCN(feature, edge, f_feature, f_edge, cl)
        pred1 = self.MLP(torch.cat((feature1, feature2), dim=1)[dateset_index])

        if iftrain:
            return pred1, cl_loss1, d, p
        return pred1


def init(i):
    if isinstance(i, nn.Linear):
        torch.nn.init.xavier_uniform_(i.weight)
