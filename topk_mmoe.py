import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import wandb

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


class Dispatcher(object):
    # gates是一个(batch_size,n_expert)的张量，表示batch内数据在各expert上的权重
    def __init__(self, n_expert, gates):
        self.gates = gates
        self.n_expert = n_expert

        nonzero_gates_index = torch.nonzero(gates)

        # nonzero_gates_index的第二列表示expert的下标，按照第二列排序，就是按照expert下标排序
        # 按照expert下标排序，是为了让每个expert接收的样本连在一起，方便构造每个expert的输入
        sorted_indices = torch.argsort(nonzero_gates_index[:, 1])
        self.nonzero_gates_index = nonzero_gates_index[sorted_indices]

        self.batch_index = self.nonzero_gates_index[:, 0]  # 第一列表示batch下标，即batch内第几个样本
        self.expert_index = self.nonzero_gates_index[:, 1]  # 第二列表示expert下标

        self.nonzero_gates = gates[self.batch_index, self.expert_index]  # 按照expert顺序排序的非零权重
        self.num_samples_per_expert = (gates > 0).sum(0).tolist()  # 每个expert接收的样本数

    def dispatch(self, x):
        # 输入为(B,d)的小批次样本
        # 输出为一个列表，列表中第i个元素是一个shape为(第i个expert接收的样本数, d)的张量
        x_expand = x[self.batch_index]
        dispatch_output = torch.split(x_expand, self.num_samples_per_expert, dim=0)  # 按照self.part_sizes分割
        return dispatch_output

    def combine(self, expert_out):
        expert_out = torch.cat(expert_out, dim=0)
        weighted_expert_out = expert_out * self.nonzero_gates.unsqueeze(1)
        zero_tensor = torch.zeros(self.gates.shape[0], expert_out.shape[1], device=expert_out.device)  # (B,d)
        combined = zero_tensor.index_add(0, self.batch_index, weighted_expert_out)
        return combined

    def expert_to_gates(self):
        # 分割出每个expert的非零权重，返回一个list，每个元素是一个shape为(第i个expert接收的样本数,)的张量
        return torch.split(self.nonzero_gates, self.num_samples_per_expert, dim=0)


class SparseMMoE(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 n_expert,
                 k,
                 n_task,
                 sparse_load_balancing_loss_coef,
                 olmo_load_balancing_loss_coef,
                 router_z_loss_coef,
                 gate_dropout,
                 expert_dropout,
                 ):

        super(SparseMMoE, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.n_expert = n_expert
        self.k = k
        self.n_task = n_task
        self.sparse_load_balancing_loss_coef = sparse_load_balancing_loss_coef
        self.olmo_load_balancing_loss_coef = olmo_load_balancing_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        self.gate_dropout = gate_dropout
        self.expert_dropout = expert_dropout

        self.experts = nn.ModuleList(
            [Expert(self.input_size, self.output_size, self.expert_dropout) for i in range(self.n_expert)])
        self.gate_dropout_layer = nn.Dropout(gate_dropout) if gate_dropout > 0 else nn.Identity()

        # 将w_gate和w_noise全部初始化为全0，保证初始时通过noise选择expert
        self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_size, n_expert), requires_grad=True) for _ in range(n_task)])
        self.b_gates = nn.ParameterList(
            [nn.Parameter(torch.zeros(n_expert), requires_grad=True) for _ in range(n_task)])
        self.w_noises = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_size, n_expert), requires_grad=True) for _ in range(n_task)])
        self.b_noises = nn.ParameterList(
            [nn.Parameter(torch.zeros(n_expert), requires_grad=True) for _ in range(n_task)])

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.normal = Normal(0.0, 1.0)  # 标准高斯分布

        assert (self.k <= self.n_expert)

    def gates_to_load(self, gates):
        """
        计算每个expert的真实负载，即接收了多少个样本(或者说在小批次数据内有多少个样本在该expert上的权重大于0)
        """
        return (gates > 0).sum(dim=0)

    # cv即coefficient of variation(变异系数), cv(x)**2 = x的方差/(x的均值**2)
    # 计算cv_squared是为了计算load balancing loss
    # 参考 https://arxiv.org/pdf/1701.06538
    def cv_squared(self, x):
        """
        The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        """
        eps = 1e-10
        # if n_expert = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)

        # 因为gates_to_load返回的是真实负载，是整数。所以这里要用float()
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def prob_in_top_k(self, clean_values, noisy_values, noise_stddev):
        """
        Computes the probability that value is in top k, given different random noise.
        Args:
        clean_values: a `Tensor` of shape [batch, n_expert].
        noisy_values: a `Tensor` of shape [batch, n_expert].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n_expert]
        Returns:
        a `Tensor` of shape [batch, n_expert].
        """

        noisy_topk_values, _ = torch.topk(noisy_values, self.k + 1)

        top_k_plus_1_values = noisy_topk_values[:, [-1]]
        prob_topk = self.normal.cdf((clean_values - top_k_plus_1_values) / noise_stddev)
        top_k_values = noisy_topk_values[:, [-2]]
        prob_after_topk = self.normal.cdf((clean_values - top_k_values) / noise_stddev)

        # 如果比top_k_plus_1_values大，就说明在topk内
        in_topk = torch.gt(noisy_values, top_k_plus_1_values)

        # 对于前k大值，除自身以外的第k大值就是第k+1大值，所以选择prob_topk
        # 对于非topk的值，除自身以外的第k大值就是第k大值，所以选择prob_after_topk
        prob = torch.where(in_topk, prob_topk, prob_after_topk)

        return prob

    def noisy_top_k_gating(self, x, train, taks_id, noise_epsilon=1e-2):
        """
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, n_expert]
            load: a Tensor with shape [n_expert]
        """
        # 引入router_z_loss，惩罚进入gating的大logit，稳定训练
        router_z_loss = torch.sum(torch.logsumexp(x, dim=-1)) / x.shape[0]

        clean_logits = x @ self.w_gates[taks_id] + self.b_gates[taks_id]

        # 只有训练时加噪音，测试时不加
        if train:
            raw_noise_stddev = x @ self.w_noises[taks_id] + self.b_noises[taks_id]
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon  # 为了数值稳定
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_k_logits, top_k_indices = logits.topk(self.k, dim=1)
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if train:
            # 这里的load_i指的是第i个expert被batch内各样本激活概率的和
            load = (self.prob_in_top_k(clean_logits, noisy_logits, noise_stddev)).sum(0)
        else:
            # 这里的load是测试时的真实负载
            load = self.gates_to_load(gates)

        gates = self.gate_dropout_layer(gates)  # 引入dropout

        return gates, load, router_z_loss

    def forward(self, x):
        load_balancing_loss = 0
        router_z_loss = 0
        outputs = []
        for i in range(self.n_task):
            gates, load, task_router_z_loss = self.noisy_top_k_gating(x, self.training, i)
            # 记录每个batch内各expert的平均门网络的输出
            for j in range(self.n_expert):
                wandb.log({f"task_{i}/expert_{j}_weight": gates[:, j].mean()})

            importance = gates.sum(0)

            # 计算Sparsely-gated MOE中的load-balancing loss
            task_sparse_load_balancing_loss = self.cv_squared(importance) + self.cv_squared(load)

            # 计算OLMO中的load-balancing loss
            load = load / x.shape[0]  # the fraction of tokens routed to one experts
            task_olmo_load_balancing_loss = self.n_expert * torch.sum(importance * load)

            # 计算load-balancing loss
            load_balancing_loss = load_balancing_loss + task_sparse_load_balancing_loss * self.sparse_load_balancing_loss_coef + \
                                  task_olmo_load_balancing_loss * self.olmo_load_balancing_loss_coef

            # 计算router_z_loss
            router_z_loss = router_z_loss + task_router_z_loss * self.router_z_loss_coef

            dispatcher = Dispatcher(self.n_expert, gates)
            expert_inputs = dispatcher.dispatch(x)
            # gates = dispatcher.expert_to_gates()
            expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.n_expert)]
            y = dispatcher.combine(expert_outputs)
            outputs.append(y)
        return outputs, load_balancing_loss, router_z_loss


class TopkMMoE(nn.Module):
    # feature_dim:输入数据的维数  expert_dim:每个专家输出的维数  n_expert:专家数量  n_task:任务数(gate数)
    def __init__(self,
                 feature_dim,
                 expert_dim,
                 n_expert,
                 n_activated_expert,
                 n_task,
                 sparse_load_balancing_loss_coef,
                 olmo_load_balancing_loss_coef,
                 router_z_loss_coef,
                 gate_dropout=0,
                 tower_dropout=0,
                 expert_dropout=0,
                 ):

        super(TopkMMoE, self).__init__()

        self.n_task = n_task
        self.sparse_mmoe = SparseMMoE(input_size=feature_dim,
                                      output_size=expert_dim,
                                      n_expert=n_expert,
                                      k=n_activated_expert,
                                      n_task=n_task,
                                      sparse_load_balancing_loss_coef=sparse_load_balancing_loss_coef,
                                      olmo_load_balancing_loss_coef=olmo_load_balancing_loss_coef,
                                      router_z_loss_coef=router_z_loss_coef,
                                      gate_dropout=gate_dropout,
                                      expert_dropout=expert_dropout,
                                      )

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

        towers_input, load_balancing_loss, router_z_loss = self.sparse_mmoe(x)

        outputs = []
        for i in range(self.n_task):
            outputs.append(self.sigmoid(self.towers[i](towers_input[i])))

        return outputs, load_balancing_loss, router_z_loss