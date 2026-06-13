import torch
import inspect
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import types

# Import modules from our codebase
import data.data_utils as data_utils
import moe_model_files.compute_graph_patcher as compute_graph_patcher


def load_model(full_model_name, quantize=False):
    model_name = full_model_name.split("/")[1]
    d_map = {"": 1} if torch.cuda.device_count() > 1 else "auto"
    d_map_judge = {"": 0} if torch.cuda.device_count() > 1 else "auto"

    if model_name == "Phi-3.5-MoE-instruct":
        if quantize:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                full_model_name,
                dtype="auto",
                device_map=d_map,
                quantization_config=quant_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=False,
            ).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                full_model_name,
                dtype="auto",
                device_map=d_map,
                attn_implementation="flash_attention_2",
                trust_remote_code=False,
            ).eval()
    elif model_name == "gpt-oss-20b":
        model = AutoModelForCausalLM.from_pretrained(
            full_model_name,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        ).eval()
    elif model_name == "Hunyuan-A13B-Instruct":
        model = AutoModelForCausalLM.from_pretrained(
            full_model_name,
            dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=False,
        ).eval()
    elif model_name == "Llama-Guard-3-8B":
        model = AutoModelForCausalLM.from_pretrained(
            full_model_name,
            dtype="auto",
            device_map=d_map_judge,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            full_model_name,
            dtype="auto",
            device_map=d_map,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).eval()

    if model_name == "deepseek-moe-16b-chat":
        from transformers.cache_utils import DynamicCache

        # Fix 'seen_tokens' (redirects to get_seq_length)
        if not hasattr(DynamicCache, "seen_tokens"):

            @property
            def seen_tokens(self):
                return self.get_seq_length()

            DynamicCache.seen_tokens = seen_tokens

        # Fix 'get_max_length' (returns None because DynamicCache grows indefinitely)
        if not hasattr(DynamicCache, "get_max_length"):

            def get_max_length(self):
                return None

            DynamicCache.get_max_length = get_max_length

        # Fix 'get_usable_length' (handles both single and double argument calls)
        if not hasattr(DynamicCache, "get_usable_length"):

            def get_usable_length(self, input_seq_len, layer_idx=0):
                # We ignore input_seq_len because DynamicCache doesn't have a fixed window
                # We just return the current length of the cache for that layer
                return self.get_seq_length(layer_idx)

            DynamicCache.get_usable_length = get_usable_length
        print("\nDynamicCache successfully patched for compatibility with deepseek-moe-16b-chat.")

        for layer in model.model.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                layer.mlp.gate.forward = types.MethodType(compute_graph_patcher.deepseek_moe_gate_forward, layer.mlp.gate)
    elif model_name == "Qwen1.5-MoE-A2.7B-Chat":
        for layer in model.model.layers:
            if hasattr(layer, "mlp"):
                layer.mlp.forward = types.MethodType(compute_graph_patcher.qwen1_5_moe_forward, layer.mlp)
    elif model_name == "gpt-oss-20b":
        for layer in model.model.layers:
            if hasattr(layer, "mlp"):
                layer.mlp.forward = types.MethodType(compute_graph_patcher.GptOssMLP_forward, layer.mlp)
                if hasattr(layer.mlp, "router"):
                    layer.mlp.router.forward = types.MethodType(compute_graph_patcher.GptOssTopKRouter_forward, layer.mlp.router)

                # Replace experts
                if hasattr(layer.mlp, "experts"):
                    old_experts = layer.mlp.experts

                    # Create new experts with per-expert gate-up layers
                    new_experts = compute_graph_patcher.GptOssExperts(model.config)
                    new_experts = new_experts.to(old_experts.down_proj.device)
                    new_experts = new_experts.to(old_experts.down_proj.dtype)

                    # Copy weights for down projection
                    new_experts.down_proj.data.copy_(old_experts.down_proj.data)
                    new_experts.down_proj_bias.data.copy_(old_experts.down_proj_bias.data)

                    # Copy weights for each expert's gate-up layer
                    for i in range(new_experts.num_experts):
                        new_experts.gate_up_layers[i].gate_up_proj.data.copy_(old_experts.gate_up_proj.data[i])
                        new_experts.gate_up_layers[i].gate_up_proj_bias.data.copy_(old_experts.gate_up_proj_bias.data[i])

                    # Replace the experts module
                    layer.mlp.experts = new_experts

        print("Patching finished")

    tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        # We can set pad_token as the eos_token or add a new one
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def generate_output(model, model_name, tokenizer, prompts, batch_size):
    if model_name in [
        "gpt-oss-20b",
        "Hunyuan-A13B-Instruct",
    ]:
        max_new_tokens = 1024  # We give thinking models more token budget
    else:
        max_new_tokens = 128

    all_outputs = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    model.eval()
    with torch.inference_mode():
        for batch_prompts in tqdm(data_utils.batchify(prompts, batch_size), total=total_batches, desc="Generating Responses"):
            input_tokens = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

            if "token_type_ids" in input_tokens:
                forward_args = inspect.signature(model.forward).parameters
                if "token_type_ids" not in forward_args:
                    input_tokens.pop("token_type_ids")

            input_ids = input_tokens["input_ids"]

            output_ids = model.generate(
                **input_tokens,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Slice out the prompt to isolate generated tokens
            generated_tokens = [output[ids.shape[-1] :] for ids, output in zip(input_ids, output_ids["sequences"])]

            # Decode
            responses = [tokenizer.decode(generated_tokens[i], skip_special_tokens=True).strip() for i in range(len(generated_tokens))]
            all_outputs.extend(responses)

            for r in responses:
                print(f"Generated response: {[r]}", flush=True)

    return all_outputs


def generate_output_sorted(model, model_name, tokenizer, prompts, batch_size):
    if model_name in [
        "gpt-oss-20b",
        "Hunyuan-A13B-Instruct",
    ]:
        max_new_tokens = 1024
    else:
        max_new_tokens = 128

    # 1. Sort by length, keeping original indices
    indexed_prompts = sorted(enumerate(prompts), key=lambda x: len(x[1]))

    # Pre-allocate array for final outputs
    all_outputs = [None] * len(prompts)
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    model.eval()
    with torch.inference_mode():
        # 2. Pass the SORTED tuples (original_idx, prompt_text) into your batchify function
        for batch_data in tqdm(data_utils.batchify(indexed_prompts, batch_size), total=total_batches, desc="Generating Responses"):

            # Unpack the batch into separate lists for indices and text
            batch_indices = [item[0] for item in batch_data]
            batch_texts = [item[1] for item in batch_data]

            input_tokens = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

            if "token_type_ids" in input_tokens:
                forward_args = inspect.signature(model.forward).parameters
                if "token_type_ids" not in forward_args:
                    input_tokens.pop("token_type_ids")

            input_ids = input_tokens["input_ids"]

            output_ids = model.generate(
                **input_tokens,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Slice out the prompt to isolate generated tokens
            generated_tokens = [output[ids.shape[-1] :] for ids, output in zip(input_ids, output_ids["sequences"])]

            # Decode
            responses = [tokenizer.decode(generated_tokens[k], skip_special_tokens=True).strip() for k in range(len(generated_tokens))]

            # 3. Use the original indices to place responses in the correct order
            for original_idx, response in zip(batch_indices, responses):
                all_outputs[original_idx] = response
                print(f"Generated response: {[response]}", flush=True)

    return all_outputs


def moderate(model, tokenizer, prompt):
    inputs = tokenizer.apply_chat_template(prompt, padding=True, truncation=True, return_tensors="pt", return_dict=True).to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.pad_token_id, do_sample=False)

    prompt_len = inputs["input_ids"].shape[-1]

    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


def batched_moderate(model, tokenizer, batch_prompts):
    inputs = tokenizer.apply_chat_template(batch_prompts, padding=True, truncation=True, return_tensors="pt", return_dict=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.pad_token_id, do_sample=False)

    prompt_len = inputs.input_ids.shape[-1]

    return tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
