def log_message(message):
    print(f"[LOG] {message}")

def calculate_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0

def save_model(model, filepath):
    import torch
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    import torch
    model.load_state_dict(torch.load(filepath))
    return model

def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)