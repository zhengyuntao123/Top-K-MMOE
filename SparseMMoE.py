# Reference: https://github.com/davidmrau/mixture-of-experts
# I have made a lot of modifications based on the above code

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, expert_dropout):  # input_dim代表输入维度，output_dim代表输出维度
        super(Expert, self).__init__()

        expert_hidden_layers = [64, 32]
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
    # gates是一个(batch_size,num_experts)的张量，表示batch内数据在各expert上的权重
    def __init__(self, num_experts, gates):
        self.gates = gates
        self.num_experts = num_experts

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
    def __init__(self, input_size, output_size, num_experts, n_task, expert_dropout=0.1, noisy_gating=True, k=2):
        super(SparseMMoE, self).__init__()
        self.noisy_gating = noisy_gating  # 在gate权重上加入噪音, 可以打破平局时的均衡
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self.expert_dropout=expert_dropout
        self.n_task=n_task
        self.k = k
        self.experts = nn.ModuleList([Expert(self.input_size, self.output_size, self.expert_dropout) for i in range(self.num_experts)])
        # 将w_gate和w_noise全部初始化为全0，保证初始时通过noise选择expert
        self.w_gates = nn.ParameterList([nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True) for _ in range(n_task)])
        self.b_gates = nn.ParameterList([nn.Parameter(torch.zeros(num_experts), requires_grad=True) for _ in range(n_task)])
        self.w_noises = nn.ParameterList([nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True) for _ in range(n_task)])
        self.b_noises = nn.ParameterList([nn.Parameter(torch.zeros(num_experts), requires_grad=True) for _ in range(n_task)])

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.normal=Normal(0.0,1.0)
        assert (self.k <= self.num_experts)

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
        # if num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)

        # 因为gates_to_load返回的是真实负载，是整数。所以这里要用float()
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def gates_to_load(self, gates):
        """
        计算每个expert的真实负载，即接收了多少个样本(或者说在小批次数据内有多少个样本在该expert上的权重大于0)
        """
        return (gates > 0).sum(dim=0)

    def prob_in_top_k(self, clean_values, noisy_values, noise_stddev):
        """
        Computes the probability that value is in top k, given different random noise.
        Args:
        clean_values: a `Tensor` of shape [batch, num_epxerts].
        noisy_values: a `Tensor` of shape [batch, num_epxerts].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, num_epxerts]
        Returns:
        a `Tensor` of shape [batch, num_epxerts].
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
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gates[taks_id] + self.b_gates[taks_id]
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noises[taks_id] + self.b_noises[taks_id]
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon  # 为了数值稳定
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_k_logits, top_k_indices = logits.topk(self.k, dim=1)
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            # 这里的load_i指的是第i个expert被batch内各样本激活概率的和
            load = (self.prob_in_top_k(clean_logits, noisy_logits, noise_stddev)).sum(0)
        else:
            # 这里的load是真实负载
            load = self.gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        load_balancing_loss=0
        outputs=[]
        for i in range(self.n_task):
            gates, load = self.noisy_top_k_gating(x, self.training, i)
            # 记录每个batch内第一个样本的门网络的输出
            for i in range(self.n_task):
                for j in range(self.num_experts):
                    writer.add_scalar(f"task_{i}/expert_{j}_weight", gates[0][j], tot_iters)

            # calculate importance loss
            importance = gates.sum(0)
            # load_balancing_loss = importance_loss + load_loss
            loss = self.cv_squared(importance) + self.cv_squared(load)
            loss *= loss_coef
            load_balancing_loss=load_balancing_loss+loss

            dispatcher = Dispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)
            # gates = dispatcher.expert_to_gates()
            expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
            y = dispatcher.combine(expert_outputs)
            outputs.append(y)
        return outputs, load_balancing_loss
