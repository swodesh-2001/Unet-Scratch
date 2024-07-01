import torch
import torchvision
from dataset import CustomDataset
from torch.utils.data import DataLoader
 

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
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

def check_accuracy( loader, model, device = "cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval() # No gradient will be saved like this. Done during evaluation
    
    with torch.no_grad():
        for X,y in loader:
            X = X.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(X))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += ( 2 * (preds * y).sum()) / ( (preds + y).sum() + 1e-8 )

    print(
        f"Got {num_correct}/{num_pixels}"
    )

    print(f"Dice Score : {dice_score/len(loader)}")
    model.train() # setting the model back to training 



def save_predictions_as_imgs(
        loader, model, folder = "saved_images/", device = "cuda"
):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device = device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()

