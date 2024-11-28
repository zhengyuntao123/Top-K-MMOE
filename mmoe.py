import torch.nn as nn
import numpy as np
import wandb
import torch

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, expert_dropout):  # input_dim代表输入维度，output_dim代表输出维度
        super(Expert, self).__init__()

        expert_hidden_layers = [16, 8]
        self.expert_layer = nn.Sequential(
            nn.Linear(input_dim, expert_hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(expert_hidden_layers[0], expert_hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(expert_hidden_layers[1], output_dim),
            nn.ReLU(),
            nn.Dropout(expert_dropout)
        )

    def forward(self, x):
        out = self.expert_layer(x)
        return out


class Expert_Gate(nn.Module):
    # feature_dim:输入数据的维数  expert_dim:每个神经元输出的维数  n_expert:专家数量  n_task:任务数(gate数)  use_gate：是否使用门控，如果不使用则各个专家取平均
    def __init__(self, feature_dim, expert_dim, n_expert, n_task, expert_dropout, gate_dropout, use_gate=True):
        super(Expert_Gate, self).__init__()
        self.n_task = n_task
        self.use_gate = use_gate
        self.n_expert = n_expert

        '''专家网络'''
        for i in range(n_expert):
            setattr(self, "expert_layer" + str(i + 1), Expert(feature_dim, expert_dim, expert_dropout))
        self.expert_layers = [getattr(self, "expert_layer" + str(i + 1)) for i in range(n_expert)]  # 为每个expert创建一个DNN

        '''门控网络'''
        for i in range(n_task):
            setattr(self, "gate_layer" + str(i + 1), nn.Sequential(nn.Linear(feature_dim, n_expert),
                                                                   nn.Softmax(dim=1),
                                                                   nn.Dropout(gate_dropout)))
        self.gate_layers = [getattr(self, "gate_layer" + str(i + 1)) for i in range(n_task)]  # 为每个gate创建一个lr+softmax

    def forward(self, x):
        if self.use_gate:
            # 多个专家网络的输出
            E_net = [expert(x) for expert in self.expert_layers]
            E_net = torch.cat(([e[:, np.newaxis, :] for e in E_net]), dim=1)  # (b,n_expert,expert_dim)

            # 多个门网络的输出
            gate_net = [gate(x) for gate in self.gate_layers]  # n_task个(b,n_expert)

            # 记录每个batch内各expert的平均门网络的输出
            for i in range(self.n_task):
                for j in range(self.n_expert):
                    wandb.log({f"task_{i}/expert_{j}_weight": gate_net[i][:, j].mean()})

            # towers计算：对应的门网络乘上所有的专家网络
            towers_input = []
            for i in range(self.n_task):
                g = gate_net[i].unsqueeze(2)  # (b,n_expert,1)
                tower_input = torch.matmul(E_net.transpose(1, 2), g)  # (b,d,n_expert)@(b,n_expert,1)-->(b,expert_dim,1)
                towers_input.append(tower_input.squeeze(2))  # (b, expert_dim)
        else:
            E_net = [expert(x) for expert in self.expert_layers]
            towers_input = sum(E_net) / len(E_net)
        return towers_input


class MMoE(nn.Module):
    # feature_dim:输入数据的维数  expert_dim:每个神经元输出的维数  n_expert:专家数量  n_task:任务数(gate数)
    def __init__(self, feature_dim, expert_dim, n_expert, n_task, use_gate=True, tower_dropout=0, expert_dropout=0,
                 gate_dropout=0):
        super(MMoE, self).__init__()

        self.n_task = n_task
        self.use_gate = use_gate
        self.Expert_Gate = Expert_Gate(feature_dim=feature_dim, expert_dim=expert_dim, n_expert=n_expert, n_task=n_task,
                                       expert_dropout=expert_dropout, gate_dropout=gate_dropout, use_gate=use_gate)

        # 对于离散变量做embedding
        # 25种离散变量
        vocab_size_list = [9, 24, 15, 5, 10, 2, 3, 6, 1, 6, 6, 50, 37, 8, 9, 8, 9, 3, 3, 5, 40, 40, 41, 5, 3]
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=i, embedding_dim=4) for i in vocab_size_list
        ])
        # 合计送入expert的维度为:25*4+14=114

        # 顶层的任务塔
        hidden_layer1 = [8, 4]
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, hidden_layer1[0]),
                nn.ReLU(),
                nn.Linear(hidden_layer1[0], hidden_layer1[1]),
                nn.ReLU(),
                nn.Dropout(tower_dropout),
                nn.Linear(hidden_layer1[1], 1))
            for i in range(n_task)
        ])

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_d, x_c):
        temp = []
        for i in range(len(self.embeddings)):
            temp.append(self.embeddings[i](x_d[:, i]))
        temp = temp + [x_c]
        x = torch.cat(temp, dim=-1)

        towers_input = self.Expert_Gate(x)

        outputs = []
        if self.use_gate:
            for i in range(self.n_task):
                outputs.append(self.sigmoid(self.towers[i](towers_input[i])))
        else:
            for i in range(self.n_task):
                outputs.append(self.sigmoid(self.towers[i](towers_input)))

        return outputs