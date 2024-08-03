import torch
import torchvision
from dataset import CustomDataset
from torch.utils.data import DataLoader
 

def save_checkpoint(state, model_name = "my_checkpoint.pth.tar"):
    assert model_name.endswith(".tar"), "model_name should end with '.pt.tar' or '.pth.tar'"
    torch.save(state, model_name)

def load_checkpoint(checkpoint_path, model):
    assert checkpoint_path.endswith(".tar") , "model_name should end with '.pt.tar' or '.pth.tar'"
    print("=> Loading Checkpoint")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
        TRAIN_DIR,
        VAL_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY = True,
):
    train_dataset = CustomDataset(
        data_dir=  TRAIN_DIR,
        transform= train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size= BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory= PIN_MEMORY,
        shuffle= True,
    )

    val_dataset = CustomDataset(
        data_dir = VAL_DIR, 
        transform= val_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size= BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory= PIN_MEMORY,
        shuffle= False,
    )

    return train_loader, val_loader

 

 