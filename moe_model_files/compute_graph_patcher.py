import torch.nn.functional as F
import torch
import torch.nn as nn


def deepseek_moe_gate_forward(self, hidden_states):
    bsz, seq_len, h = hidden_states.shape
    # Compute gating score
    hidden_states = hidden_states.view(-1, h)
    logits = F.linear(hidden_states, self.weight, None)
    if self.scoring_func == "softmax":
        scores = logits.softmax(dim=-1)
    else:
        raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

    # Select top-k experts
    topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

    # Norm gate to sum 1
    if self.top_k > 1 and self.norm_topk_prob:
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator

    # Expert-level computation auxiliary loss
    if self.training and self.alpha > 0.0:
        scores_for_aux = scores
        aux_topk = self.top_k
        # Always compute aux loss based on the naive greedy topk method
        topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
        if self.seq_aux:
            scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
            ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
            ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                seq_len * aux_topk / self.n_routed_experts
            )
            aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
        else:
            mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
            ce = mask_ce.float().mean(0)
            Pi = scores_for_aux.mean(0)
            fi = ce * self.n_routed_experts
            aux_loss = (Pi * fi).sum() * self.alpha
    else:
        aux_loss = logits
    return topk_idx, topk_weight, aux_loss


def qwen1_5_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)
    shared_expert_logits = self.shared_expert_gate(hidden_states)

    routing_weights = torch.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # We cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)

    expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hitted:
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    shared_expert_output = self.shared_expert(hidden_states)
    shared_expert_output = torch.sigmoid(shared_expert_logits) * shared_expert_output

    final_hidden_states = final_hidden_states + shared_expert_output

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def GptOssTopKRouter_forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)
    return router_logits


def GptOssMLP_forward(self, hidden_states):
    router_logits = self.router(hidden_states)  # (num_experts, seq_len)
    router_top_value, router_indices = torch.topk(router_logits, 4, dim=-1)  # (seq_len, top_k)
    router_top_value = torch.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
    router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
    routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
    return routed_out, router_scores


class GptOssGateUpLayer(nn.Module):
    def __init__(self, hidden_size, expert_dim):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(hidden_size, 2 * expert_dim))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(2 * expert_dim))

    def forward(self, hidden_states):
        gate_up = hidden_states @ self.gate_up_proj + self.gate_up_proj_bias
        return gate_up[..., ::2], gate_up[..., 1::2]


class GptOssExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size

        # One GptOssGateUpLayer per expert
        self.gate_up_layers = nn.ModuleList([GptOssGateUpLayer(self.hidden_size, self.expert_dim) for _ in range(self.num_experts)])

        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.expert_dim, self.hidden_size))
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))

        self.alpha = 1.702
        self.limit = 7.0

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        num_experts = routing_weights.shape[1]

        if self.training:
            next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with torch.no_grad():
                expert_mask = F.one_hot(router_indices, num_classes=num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_idx in expert_hit[:]:
                expert_id = expert_idx[0].item()
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_id])

                current_state = hidden_states[token_idx]
                gate, up = self.gate_up_layers[expert_id](current_state)

                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)
                gated_output = (up + 1) * glu

                out = gated_output @ self.down_proj[expert_id] + self.down_proj_bias[expert_id]
                weighted_output = out[0] * routing_weights[token_idx, expert_id, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))

            next_states = next_states.view(batch_size, -1, self.hidden_size)

        else:
            hidden_states = hidden_states.repeat(num_experts, 1)
            hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)

            gate, up = [], []
            for i in range(num_experts):
                g, u = self.gate_up_layers[i](hidden_states[i])
                gate.append(g)
                up.append(u)
            gate = torch.stack(gate, dim=0)
            up = torch.stack(up, dim=0)

            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)

            next_states = torch.bmm((up + 1) * glu, self.down_proj)
            next_states = next_states + self.down_proj_bias[..., None, :]
            next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
            next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
            next_states = next_states.sum(dim=0)

        return next_states