from dataclasses import dataclass


@dataclass
class MoEModelConfig:
    model_name: str
    gate_name: str
    top_k: int
    num_router_expert: int


models = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": MoEModelConfig(
        model_name="Qwen3-30B-A3B-Instruct-2507",
        gate_name="mlp.gate",
        top_k=8,
        num_router_expert=128,
    ),
    "microsoft/Phi-3.5-MoE-instruct": MoEModelConfig(
        model_name="Phi-3.5-MoE-instruct",
        gate_name="block_sparse_moe.gate",
        top_k=2,
        num_router_expert=16,
    ),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": MoEModelConfig(
        model_name="Mixtral-8x7B-Instruct-v0.1",
        gate_name="block_sparse_moe.gate",
        top_k=2,
        num_router_expert=8,
    ),
    "openai/gpt-oss-20b": MoEModelConfig(
        model_name="gpt-oss-20b",
        gate_name="mlp.router",
        top_k=4,
        num_router_expert=32,
    ),
    "Qwen/Qwen1.5-MoE-A2.7B-Chat": MoEModelConfig(
        model_name="Qwen1.5-MoE-A2.7B-Chat",
        gate_name="mlp.gate",
        top_k=4,
        num_router_expert=60,
    ),
    "tencent/Hunyuan-A13B-Instruct": MoEModelConfig(
        model_name="Hunyuan-A13B-Instruct",
        gate_name="mlp.gate.wg",
        top_k=8,
        num_router_expert=64,
    ),
    "deepseek-ai/deepseek-moe-16b-chat": MoEModelConfig(
        model_name="deepseek-moe-16b-chat",
        gate_name="mlp.gate",
        top_k=6,
        num_router_expert=64,
    ),
}
