from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel_experts import ParallelExperts, compute_gating

from .gate import top_k_gating


class MoE(nn.Module):
    """
    A Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.

    Args:
        input_size: integer - size of the input
        hidden_size: integer - size of the expert's hidden layer
        num_experts: an integer - number of experts
        top_k: an integer - how many experts to use for each batch element
        bias: a boolean - whether to include bias in linear layers
        activation: an activation function to apply to expert's outputs
        glu: an boolean - whether to use GLU activation
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_experts,
        top_k,
        bias=True,
        activation=None,
        glu=True,
    ):
        super(MoE, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.glu = glu
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(input_size))
            torch.nn.init.zeros_(self.bias)
        else:
            self.bias = None

        self.input_linear = ParallelExperts(
            num_experts, input_size, hidden_size * 2 if glu else hidden_size)
        self.output_linear = ParallelExperts(
            num_experts, hidden_size, input_size)

        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

        self.router = top_k_gating(
            input_size=input_size,
            num_experts=num_experts,
            top_k=top_k,
        )

    def extra_repr(self):
        return "k={}, e={}".format(self.top_k, self.num_experts)

    def get_aux_loss_and_clear(self):
        """
        Get the accumulated auxiliary loss and clear it.

        Returns:
            float: Accumulated auxiliary loss.
        """

        return self.gate.get_aux_loss_and_clear()

    def compute_gate(self, x):
        top_k_indices, self.top_k_gates = self.router(x)

        self.batch_gates, self.batch_index, expert_size, self.index_sorted_experts = compute_gating(
            self.top_k, self.num_experts, self.top_k_gates, top_k_indices
        )
        self.expert_size = expert_size.tolist()

        return self.router.loss

    def batch_forward(self, x):
        """
        Forward pass of the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.
            skip_mask (Tensor): Skip mask tensor.
            sample_topk (int): Number of experts to sample during training.
            multiply_by_gates (bool): Whether to multiply outputs by gating values.

        Returns:
            Tensor: Output tensor.
            float: Gating loss.
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        loss = self.compute_gate(x)

        expert_inputs = x[self.batch_index]
        h = self.input_linear(expert_inputs, self.expert_size)
        if self.glu:
            h, g = h.chunk(2, dim=-1)
            h = self.activation(h) * g
        else:
            h = self.activation(h)
        expert_outputs = self.output_linear(h, self.expert_size)

        # 将专家输出乘以对应的门控值。
        expert_outputs = expert_outputs * self.batch_gates[:, None]

        # 使用 index_add 方法将乘以门控值的专家输出张量根据 batch_index 分散到全零张量 zeros 中，得到混合输出张量 y。
        zeros = torch.zeros((bsz * length, self.input_size),
                            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)

        # 将混合输出张量 y 重新整形为三维张量，形状与输入张量 x 相同。
        y = y.view(bsz, length, self.input_size)
        # 如果设置了偏置项，则将偏置项添加到输出张量 y 中。
        if self.bias is not None:
            y = y + self.bias
        return y, loss

    def single_forward(self, x):
        bsz, length, emb_size = x.size()

        x = x.reshape(1, self.input_size)
        top_k_indices, top_k_gates = self.router(x)
        loss = self.router.loss

        y_list = []
        for i in range(self.top_k):
            expert_idx = top_k_indices[0, i]

            h = F.linear(x, self.input_linear.weight[expert_idx])
            if self.glu:
                h, g = h.chunk(2, dim=-1)
                h = self.activation(h) * g
            else:
                h = self.activation(h)
            y = F.linear(
                h, self.output_linear.weight[expert_idx]) * top_k_gates[0, i]

            y_list.append(y)

        y = sum(y_list)
        y = y.view(bsz, length, self.input_size)
        if self.bias is not None:
            y = y + self.bias
        return y, loss

    def forward(self, x):
        """
        Forward pass of the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        bsz, length, emb_size = x.size()
        if bsz * length == 1:
            return self.single_forward(x)
        else:
            return self.batch_forward(x)

    def single_map(self, x):
        bsz, length, emb_size = x.size()

        x = x.reshape(1, self.input_size)
        self.top_k_indices, self.top_k_gates = self.router(x)
        loss = self.router.loss

        y_list = []
        for i in range(self.top_k):
            expert_idx = self.top_k_indices[0, i]
            y = F.linear(x, self.input_linear.weight[expert_idx])
            y_list.append(y)
        y = torch.cat(y_list, dim=0)
        y = y.view(bsz, length, self.top_k, -1)
        return y, loss

    def batch_map(self, x):
        """

        Args:
            x: tensor shape [batch_size, input_size]
            train: a boolean scalar.
            loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
            y: a tensor with shape [batch_size, output_size].
            extra_training_loss: a scalar.  This should be added into the overall
            training loss of the model.  The backpropagation of this loss
            encourages all experts to be approximately equally used across a batch.
        """
        """
        Map input through the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.
            skip_mask (Tensor): Skip mask tensor.
            sample_topk (int): Number of experts to sample during training.
            return_indices (bool): Whether to return expert indices.

        Returns:
            Tensor: Output tensor.
            float: Gating loss.
        """
        # 解析输入张量的形状，获取批次大小（bsz）、序列长度（length）和输入特征维度（emb_size）。
        bsz, length, emb_size = x.size()
        # 将输入张量 x 重新整形为二维张量，形状为 (bsz * length, emb_size)，以便进行批次级别的处理。
        x = x.reshape(-1, emb_size)
        # 调用 compute_gate 方法计算门控损失。
        loss = self.compute_gate(x)

        # 根据 batch_index 提取每个样本所属的专家输入，形状为 (num_experts, expert_size)。
        expert_inputs = x[self.batch_index]
        # 将专家输入传递给 input_linear 层，使用专家大小信息进行线性变换，得到专家输出。
        expert_outputs = self.input_linear(expert_inputs, self.expert_size)

        # 创建一个全零张量 zeros，形状为 (bsz * length * top_k, hidden_size)，数据类型和设备与 expert_outputs 相同。
        zeros = torch.zeros(
            (bsz * length * self.top_k, self.hidden_size), dtype=expert_outputs.dtype, device=expert_outputs.device
        )
        # 使用 index_add 方法将专家输出根据 index_sorted_experts 分散到全零张量 zeros 中，得到混合输出张量 y。
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        # 将混合输出张量 y 重新整形为四维张量，形状为(bsz, length, top_k, hidden_size)。
        y = y.view(bsz, length, self.top_k, -1)
        return y, loss

    def map(self, x):
        """
        Map input through the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        bsz, length, emb_size = x.size()
        if bsz * length == 1:
            return self.single_map(x)
        else:
            return self.batch_map(x)

    def single_reduce(self, x):
        bsz, length, k, emb_size = x.size()

        x = x.reshape(k, emb_size)

        y_list = []
        for i in range(self.top_k):
            expert_idx = self.top_k_indices[0, i]
            y = F.linear(x[i], self.output_linear.weight[expert_idx]
                         ) * self.top_k_gates[0, i]
            y_list.append(y)
        y = sum(y_list)
        y = y.view(bsz, length, self.input_size)
        if self.bias is not None:
            y = y + self.bias
        return y

    def batch_reduce(self, x):
        """
        Reduce the mapped output.

        Args:
            x (Tensor): Mapped output tensor.
            multiply_by_gates (bool): Whether to multiply outputs by gating values.

        Returns:
            Tensor: Reduced output tensor.
        """

        # 解析输入张量的形状，获取批次大小（bsz）、序列长度（length）、专家数量（k）和嵌入维度（emb_size）。
        bsz, length, k, emb_size = x.size()
        # 将输入张量 x 重新整形为二维张量，形状为 (bsz * length * k, emb_size)。
        x = x.reshape(-1, emb_size)

        # 根据 index_sorted_experts 提取每个样本所属的专家输入，形状为 (num_experts, expert_size)。
        expert_inputs = x[self.index_sorted_experts]
        # 将专家输入传递给 output_linear 层，使用专家大小信息进行线性变换，得到专家输出。
        expert_outputs = self.output_linear(expert_inputs, self.expert_size)

        # 将专家输出乘以对应的门控值。
        expert_outputs = expert_outputs * self.batch_gates[:, None]

        # 创建一个全零张量 zeros，形状为 (bsz * length, input_size)，数据类型和设备与 expert_outputs 相同。
        zeros = torch.zeros((bsz * length, self.input_size),
                            dtype=expert_outputs.dtype, device=expert_outputs.device)
        # 使用 index_add 方法将乘以门控值的专家输出张量根据 batch_index 分散到全零张量 zeros 中，得到降维后的输出张量 y。
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        # 将降维后的输出张量 y 重新整形为三维张量，形状为 (bsz, length, input_size)。
        y = y.view(bsz, length, self.input_size)
        # 如果设置了偏置项，则将偏置项添加到输出张量 y 中。
        if self.bias is not None:
            y = y + self.bias
        return y

    def reduce(self, x):
        """
        Reduce the mapped output.

        Args:
            x (Tensor): Mapped output tensor.

        Returns:
            Tensor: Reduced output tensor.
        """
        bsz, length, k, emb_size = x.size()
        if bsz * length == 1:
            return self.single_reduce(x)
        else:
            return self.batch_reduce(x)


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
