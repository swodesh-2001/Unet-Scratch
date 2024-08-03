import argparse
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
import engine,models,utils 
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description=" Script for UNET From Scratch")

    # Hyperparameters
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--image_height', type=int, default=160, help='Height of the input images')
    parser.add_argument('--image_width', type=int, default=240, help='Width of the input images')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Pin memory for DataLoader')
    parser.add_argument('--load_model', type=bool, default= True, help='Load pre-trained model')
    parser.add_argument('--verbose', type=bool, default= True, help='Whether to display training info or not')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader')
    parser.add_argument('--load_model_path', type=str, default='./models/final.pth.tar', help='Path to the pre-trained model')

    # Directories
    parser.add_argument('--train_dir', type=str, default='./microscope_data/train', help='Directory for training data')
    parser.add_argument('--test_dir', type=str, default='./microscope_data/validation', help='Directory for validation data')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    IMAGE_HEIGHT = args.image_height
    IMAGE_WIDTH = args.image_width
    PIN_MEMORY = args.pin_memory
    LOAD_MODEL = args.load_model
    NUM_WORKERS = args.num_workers
    LOAD_MODEL_PATH = args.load_model_path
    VERBOSE = args.verbose
    TRAIN_DIR = args.train_dir
    TEST_DIR = args.test_dir

    writer = SummaryWriter(log_dir='./logs')
    train_transform = A.Compose(

        [
            A.Resize( height= IMAGE_HEIGHT, width = IMAGE_WIDTH ) ,
            A.Rotate(limit = 35, p = 1.0),
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.1),
            A.Normalize(
                mean = [0.0,0.0,0.0],
                std = [1.0,1.0,1.0],
                max_pixel_value = 255.0

            ),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose(
        [
            A.Resize( height= IMAGE_HEIGHT, width = IMAGE_WIDTH ) ,
            A.Normalize(
                mean = [0.0,0.0,0.0],
                std = [1.0,1.0,1.0],
                max_pixel_value = 255.0

            ),
            ToTensorV2(),
        ]
    )
    train_loader, test_loader = utils.get_loaders(

        TRAIN_DIR,
        TEST_DIR,
        BATCH_SIZE,
        train_transform,
        test_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    model = models.UNET(in_channels=3 , out_channels=1).to(device)

    if LOAD_MODEL :
        utils.load_checkpoint(LOAD_MODEL_PATH, model)
    
    loss_fn = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr= LEARNING_RATE)
    results = engine.train(model=model,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                loss_fn=loss_fn,
                verbose= VERBOSE, 
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)
    
    for i, (train_loss, train_acc, test_loss, test_acc) in enumerate(zip(results['train_loss'], results['train_acc'], results['test_loss'], results['test_acc'])):
        writer.add_scalar('Loss/train', train_loss, i)
        writer.add_scalar('Accuracy/train', train_acc, i)
        writer.add_scalar('Loss/valid', test_loss, i)
        writer.add_scalar('Accuracy/valid', test_acc, i)


    checkpoint = {
            "state_dict" : model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
     
    utils.save_checkpoint( checkpoint, model_name="./models/final.pth.tar")


if __name__ == "__main__" :
    main()