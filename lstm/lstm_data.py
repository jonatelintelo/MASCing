import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

class MoETraceDataset(Dataset):
    def __init__(self, traces, labels):
        self.traces = traces
        self.labels = labels

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        # Convert the float16 numpy array to a float32 PyTorch tensor for stable training
        trace_tensor = torch.tensor(self.traces[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        return trace_tensor, label_tensor

def pad_collate_fn(batch):
    # Separate traces and labels
    traces, labels = zip(*batch)
    
    # Calculate actual lengths before padding
    lengths = torch.tensor([len(t) for t in traces], dtype=torch.int64)
    
    # Pad the traces with zeros. 
    # batch_first=True makes it (Batch, Max_Seq_Len, Layers, Experts)
    padded_traces = pad_sequence(traces, batch_first=True, padding_value=0.0)
    
    # Stack the labels
    labels = torch.stack(labels).unsqueeze(1) # Shape: (Batch, 1)
    
    return padded_traces, labels, lengths

def get_dataLoader(dataset, batch_size, shuffle, collate_fn=pad_collate_fn):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
