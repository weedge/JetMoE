import torch
import torch.nn as nn
import torch.nn.functional as F


class top_k_gating(nn.Module):
    def __init__(
        self,
        input_size,
        num_experts,
        top_k,
    ):
        """
        Initialize the top-k gating mechanism.

        Args:
            input_size (int): Size of the input.
            num_experts (int): Number of experts.
            top_k (int): Number of top experts to select.
            acc_aux_loss (bool): Whether to accumulate auxiliary loss statistics.
            dropout (float): Dropout rate for gating network.
            hidden_size (int): Hidden size of the gating network.
            sample_topk (int): Number of top-k experts to sample during training.
            aux_loss (str): Type of auxiliary loss ('mi' or 'switch').
            gate_type (str): Type of gating mechanism ('mlp', 'linear', or 'gmm').
        """
        super().__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        assert top_k <= num_experts
        self.top_k = top_k

        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def extra_repr(self):
        """
        Return extra representation string for the module.
        """
        return "k={}, num_experts={}".format(self.top_k, self.num_experts)

    def compute_aux_loss(self, probs, logits, gates):
        """
        Calculate and return the auxiliary loss based on the accumulated statistics.

        Args:
            eps (float): Small epsilon value for numerical stability.

        Returns:
            torch.Tensor: The calculated auxiliary loss.
        """
        # 获取 logits 张量的批次大小，即样本数量
        count = logits.size(0)
        # 计算每个专家被选中的概率之和，即将概率沿着批次维度求和。
        # probs 是一个张量，其每个元素表示对应专家被选中的概率。
        probs = probs.sum(0)
        # 计算每个专家被选中的频率，即计算门控值大于0的次数（即专家被选中的次数），
        # 然后将其沿着批次维度求和。
        # gates 是一个张量，其每个元素表示对应专家被选中的门控值。
        freq = (gates > 0).float().sum(0)
        # 计算 logits 张量经过 softmax 处理后的平方和的对数。
        # 这里首先使用 softmax 函数将 logits 转换为概率分布，
        # 然后计算概率分布的每个样本的平方和，并取对数，最后将结果沿着批次维度求和。
        lsesq = (torch.log(torch.exp(logits).sum(dim=-1)) ** 2).sum()

        # 计算专家选择损失，其计算方式为对每个专家的概率和频率进行归一化，然后计算它们的点积，最后将结果乘以专家数量。
        switchloss = self.num_experts * \
            (F.normalize(probs, p=1, dim=0) * F.normalize(freq, p=1, dim=0)).sum()
        # 计算 z 损失，即 logits 的对数平方和除以样本数量
        zloss = lsesq / count
        # 将专家选择损失和 z 损失加权相加得到最终的辅助损失
        loss = switchloss + 0.1 * zloss

        return loss

    def forward(self, x):
        """
        Compute the top-k gating for the input.

        See paper: https://arxiv.org/abs/1701.06538.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, input_size].
            skip_mask (torch.Tensor): Skip mask tensor (binary) with the same shape as `x`.
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float

        Returns:
            torch.Tensor: Top-k indices.
            torch.Tensor: Top-k gating values.
            torch.Tensor: Probability values for each expert.
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        logits = self.layer(x).float()
        print(logits)
        # 对logits进行按行（即每个样本）的top-k操作，返回前k个最大值的对应logits和索引
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
        print(top_k_logits, top_k_indices)
        # 对top_k_logits进行softmax操作，得到对应的门控值（gates），这里的门控值表示对应的专家在预测中所占的比例
        top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(x)

        # 训练时才计算辅助loss值, 为了专家之间的负载平衡
        if self.training:
            # from: switch transformer: https://arxiv.org/pdf/2101.03961.pdf  A Differentiable Load Balancing Loss
            # 对logits进行softmax操作，得到每个类别的概率分布
            probs = torch.softmax(logits, dim=1)
            zeros = torch.zeros_like(probs)
            # Convert zeros to match top_k_gates dtype
            zeros = zeros.to(top_k_gates.dtype)
            gates = zeros.scatter(1, top_k_indices, top_k_gates)
            self.loss = self.compute_aux_loss(probs, logits, gates)
        else:
            self.loss = 0

        return top_k_indices, top_k_gates


if __name__ == "__main__":
    # Testing this out, again:
    num_experts = 8
    top_k = 2
    n_embd = 16

    input = torch.randn(2, 4, n_embd)  # Example input
    input = input.reshape(-1,n_embd)
    noisy_top_k_gate = top_k_gating(n_embd, num_experts, top_k)
    top_k_indices, top_k_gates = noisy_top_k_gate(input)
    print(top_k_gates.shape, top_k_gates)
    print(top_k_indices.shape, top_k_indices)
