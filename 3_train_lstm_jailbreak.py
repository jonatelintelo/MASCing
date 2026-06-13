import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import sys
import os

# Project Modules
import moe_model_files.model_configurations as model_configurations
import data.data_utils as data_utils
import argument_parser as argument_parser
import lstm.lstm_model as lstm_model
import lstm.lstm_data as lstm_data

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def train(traces, labels, num_total_experts):
    # --- Config ---
    BATCH_SIZE = 512
    LR = 0.01
    EPOCHS = 15

    # 1. Initialize the FULL Dataset first
    # This runs your auto-detection logic once on all data
    dataset = lstm_data.MoETraceDataset(traces, labels)

    # 2. Extract detected params NOW (Before splitting)
    # We access these from the dataset directly
    DETECTED_LAYERS = traces[0].shape[1]

    # 3. Perform the Split
    total_size = len(dataset)
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size

    # random_split wraps your dataset into two "Subset" objects
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 4. Create DataLoaders from the splits
    train_loader = lstm_data.get_dataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = lstm_data.get_dataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5. Initialize Model
    model = lstm_model.MoETraceClassifierLinear(
        num_total_experts=num_total_experts,
        num_layers=DETECTED_LAYERS,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\n🚀 Starting training.")
    print(f"📊 Data split: {len(train_dataset)} Train | {len(val_dataset)} Validation\n")

    for epoch in range(EPOCHS):

        # ==========================
        # 1. TRAINING PHASE
        # ==========================
        model.train()

        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{EPOCHS} [Train]",
        )

        for i, (x_batch, y_batch, lengths) in pbar:
            optimizer.zero_grad()

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward & Backward
            logits = model(x_batch, lengths)

            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct_preds += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)

            # Dynamic Progress Bar Update
            current_train_loss = running_loss / (i + 1)
            current_train_acc = correct_preds / total_samples

            pbar.set_postfix({"loss": f"{current_train_loss:.4f}", "acc": f"{current_train_acc:.2%}"})

        # ==========================
        # 2. VALIDATION PHASE
        # ==========================
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        # Disable gradient calculation for speed and memory efficiency
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch, lengths in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch, lengths)
                loss = criterion(logits, y_batch)

                val_running_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        # Calculate final validation metrics for the epoch
        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total

        # Print summary using tqdm.write to avoid breaking the progress bar
        tqdm.write(f"   🟢 Validation: Loss: {val_epoch_loss:.4f} | Acc: {val_epoch_acc:.2%}")
        tqdm.write("-" * 50)

    print("\n✅ Training complete.")
    return model, val_loader, criterion, DETECTED_LAYERS


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
        # LLMs
        "Qwen/Qwen3-30B-A3B-Instruct-2507",  # 0
        "microsoft/Phi-3.5-MoE-instruct",  # 1
        "mistralai/Mixtral-8x7B-Instruct-v0.1",  # 2
        "openai/gpt-oss-20b",  # 3
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",  # 4
        "tencent/Hunyuan-A13B-Instruct",  # 5
        "deepseek-ai/deepseek-moe-16b-chat",  # 6
    ]

    model_config = model_configurations.models[models[model_id]]

    print(f"\nSelected model: {model_config.model_name}")

    print("\nLoading precomputed traces...")
    lstm_input_dir = os.path.join(root_folder, "data", "lstm_input", model_config.model_name)
    traces = data_utils.load_data(os.path.join(lstm_input_dir, f"{model_config.model_name}_jailbreak_traces.pkl"))
    labels = data_utils.load_data(os.path.join(lstm_input_dir, f"{model_config.model_name}_jailbreak_labels.pkl"))

    if print_logging:
        print(f"traces length: {len(traces)}")
        print(f"labels length: {len(labels)}")
        print(f"np.array(traces[0]).shape: {np.array(traces[0]).shape}")
        print(f"np.array(labels[0]).shape: {np.array(labels[0]).shape}")
        print(f"traces[0]: {traces[0]}")
        print(f"labels[0]: {labels[0]}")
        print(f"traces[-1]: {traces[len(traces)-1]}")
        print(f"labels[-1]: {labels[len(labels)-1]}")

    trained_lstm_model, val_loader, criterion, DETECTED_LAYERS = train(traces, labels, num_total_experts=model_config.num_router_expert)

    checkpoint = {"num_total_experts": model_config.num_router_expert, "num_layers": DETECTED_LAYERS, "model_state_dict": trained_lstm_model.state_dict()}

    lstm_model_dir = os.path.join(root_folder, "lstm", "trained_lstm_models", model_config.model_name)
    os.makedirs(lstm_model_dir, exist_ok=True)

    lstm_model_path = os.path.join(lstm_model_dir, f"{model_config.model_name}_jailbreak_lstm.pkl")

    torch.save(checkpoint, lstm_model_path)

    print("\n🔍 Testing loaded model on validation set...")

    # Initialize and load the model
    checkpoint = torch.load(lstm_model_path)

    loaded_model = lstm_model.MoETraceClassifierLinear(num_total_experts=checkpoint["num_total_experts"], num_layers=checkpoint["num_layers"])

    loaded_model.load_state_dict(checkpoint["model_state_dict"])

    # Move to the same device as the original model
    device = next(trained_lstm_model.parameters()).device
    loaded_model = loaded_model.to(device)

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    loaded_model.eval()
    with torch.no_grad():
        for x_batch, y_batch, lengths in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = loaded_model(x_batch, lengths)

            loss = criterion(logits, y_batch)

            val_running_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            val_correct += (preds == y_batch).sum().item()
            val_total += y_batch.size(0)

    val_epoch_loss = val_running_loss / len(val_loader)
    val_epoch_acc = val_correct / val_total

    print(f"\nLoaded Model Validation: Loss: {val_epoch_loss:.4f} | Acc: {val_epoch_acc:.2%}")

    print("\n------------------ Job Finished ------------------")
