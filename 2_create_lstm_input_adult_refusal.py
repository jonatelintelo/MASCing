import torch
import numpy as np
import sys
import os
import inspect
from tqdm import tqdm

# Project imports
import moe_model_files.model_configurations as model_configurations
import moe_model_files.model_utils as model_utils
import data.data_utils as data_utils
import argument_parser as argument_parser


def find_token_range_by_offsets(prompt_text, question_text, offsets, print_logging):
    """Finds token start and end using character offset mappings."""
    # Find where the question starts in the raw string
    char_start = prompt_text.rfind(question_text.strip())
    if char_start == -1:
        return None, None

    char_end = char_start + len(question_text.strip())

    start_idx = None
    end_idx = None

    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:  # Skip (0,0) padding or special tokens
            continue

        # First token that overlaps with the start of the question text
        if start_idx is None and tok_end > char_start:
            start_idx = i

        # Continuously update the end_idx as long as the token starts before the question ends
        if tok_start < char_end:
            end_idx = i

    if print_logging and start_idx is not None:
        print(f"Question char range: {char_start} to {char_end}")
        print(f"start token idx: {start_idx}, end token idx: {end_idx}")

    return start_idx, end_idx


def find_token_range(question_ids, prompt_ids, print_logging):
    """Finds the start and end indices of the question within the full prompt."""
    len_q = len(question_ids)
    len_p = len(prompt_ids)
    for i in range(len_p - len_q + 1):
        if np.array_equal(prompt_ids[i : i + len_q], question_ids):
            if print_logging:
                print(f"prompt_ids: {prompt_ids}")
                print(f"question_ids: {question_ids}")
                print(f"start: {i}")
                print(f"end: {i + len_q - 1}")
            return i, i + len_q - 1
    return None, None


if __name__ == "__main__":
    arguments = argument_parser.parse_arguments()
    root_folder = arguments.root
    model_id = arguments.model_id
    print_logging = arguments.print_logging

    if print_logging:
        print(f"\nPython version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA build version: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"First GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"Test tensor on GPU: {torch.rand(5).cuda().device}")

    models = [
        "Qwen/Qwen3-30B-A3B-Instruct-2507",  # 0
        "microsoft/Phi-3.5-MoE-instruct",  # 1
        "mistralai/Mixtral-8x7B-Instruct-v0.1",  # 2
        "openai/gpt-oss-20b",  # 3
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",  # 4
        "tencent/Hunyuan-A13B-Instruct",  # 5
        "deepseek-ai/deepseek-moe-16b-chat",  # 6
    ]

    model_config = model_configurations.models[models[model_id]]
    print(f"\nInitializing: {model_config.model_name}")

    model, tokenizer = model_utils.load_model(models[model_id])  # Function laod_model already puts model on device and in .eval() mode

    questions, labels = data_utils.load_adult_refusal_dataset(root_folder, model_config.model_name, malicious_only=False)
    prompts = data_utils.construct_prompt(tokenizer, questions, model_config.model_name)

    current_batch_activations = {}

    def get_activation_hook(layer_idx):
        def hook(module, input, output):
            logits = output[2] if isinstance(output, (tuple, list)) else output
            current_batch_activations[layer_idx] = logits.detach().to(torch.float16).cpu()

        return hook

    handles = []
    layer_names = [n for n, m in model.named_modules() if n.lower().endswith(model_config.gate_name.lower())]
    for i, name in enumerate(layer_names):
        module = dict(model.named_modules())[name]
        handles.append(module.register_forward_hook(get_activation_hook(i)))

    final_traces = []
    final_labels = []
    failed_matches = 0

    BATCH_SIZE = 16
    total_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE

    # Batched Forward Pass & Logit Extraction
    print(f"\nStarting Trace Collection...")

    with torch.inference_mode():
        for b_idx, batch_prompts in enumerate(tqdm(data_utils.batchify(prompts, BATCH_SIZE), total=total_batches)):
            current_batch_activations.clear()

            # Add return_offsets_mapping=True to the tokenizer call
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)

            # Pop the offsets before sending inputs to the model
            offset_mappings = inputs.pop("offset_mapping").cpu().numpy()
            inputs = inputs.to(model.device)

            if "token_type_ids" in inputs:
                forward_args = inspect.signature(model.forward).parameters
                if "token_type_ids" not in forward_args:
                    inputs.pop("token_type_ids")

            b_size, s_len = inputs.input_ids.shape

            model(**inputs)

            for l_idx in range(len(layer_names)):
                if current_batch_activations[l_idx].dim() == 2:
                    current_batch_activations[l_idx] = current_batch_activations[l_idx].view(b_size, s_len, -1)

            input_ids_np = inputs.input_ids.cpu().numpy()

            for p_idx in range(b_size):
                global_p_idx = (b_idx * BATCH_SIZE) + p_idx

                if global_p_idx >= len(prompts):  # Changed from tokenized_questions
                    break

                prompt_text = batch_prompts[p_idx]
                q_text = questions[global_p_idx]

                if b_idx == 0 and p_idx == 0:
                    print(f"Prompt: {prompt_text}")

                # Pass the raw text and offsets instead of the token IDs
                start, end = find_token_range_by_offsets(prompt_text, q_text, offset_mappings[p_idx], print_logging if b_idx == 0 and p_idx == 0 else False)

                if start is None:
                    failed_matches += 1
                    if print_logging:
                        print(f"\nFailed to find token match for prompt index {global_p_idx}")
                    continue

                prompt_trace = []

                for l_idx in range(len(layer_names)):
                    token_probs = current_batch_activations[l_idx][p_idx, start : end + 1, :].cpu().numpy()
                    prompt_trace.append(token_probs)

                stacked_trace = np.stack(prompt_trace, axis=1)
                final_traces.append(stacked_trace)
                final_labels.append(labels[global_p_idx])

    for h in handles:
        h.remove()

    if failed_matches > 0:
        print(f"\nWarning: Could not find exact token match for {failed_matches} prompts. Check tokenizer spacing logic.")

    # Save Results
    save_path = os.path.join(root_folder, "data", "lstm_input", model_config.model_name)
    os.makedirs(save_path, exist_ok=True)

    print("\nSaving traces to disk...")
    data_utils.save_data(final_traces, os.path.join(save_path, f"{model_config.model_name}_adult_refusal_traces.pkl"))
    data_utils.save_data(final_labels, os.path.join(save_path, f"{model_config.model_name}_adult_refusal_labels.pkl"))

    if print_logging and len(final_traces) > 0:
        print(f"\nNumber of traces: {len(final_traces)}")
        print(f"Number of labels: {len(final_labels)}")
        print(f"Shape of a trace: {final_traces[0].shape}")
        print(f"traces[0]: {final_traces[0]}")
        print(f"labels[0]: {final_labels[0]}")
        print(f"traces[-1]: {final_traces[len(final_traces)-1]}")
        print(f"labels[-1]: {final_labels[len(final_labels)-1]}")

    print(f"\nSaved {len(final_traces)} traces to {save_path}")
    print("\n------------------ Job Finished ------------------")
