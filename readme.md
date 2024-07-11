# UNET FROM SCRATCH PYTHON

 

## Evaluation Metrics for Image Segmentation

For image segmentation tasks, the goal is to make the predicted mask as similar to the ground truth as possible. The most commonly used evaluation metrics for image segmentation tasks are:

### Jaccard Index (IoU)

The Jaccard Index, also known as Intersection over Union (IoU), measures the similarity between the predicted segmentation and the ground truth. It is defined as the size of the intersection divided by the size of the union of the predicted and ground truth masks.

\[ \text{IoU} = \frac{|A \cap B|}{|A \cup B|} \]

Where:
- \( A \) is the ground truth mask
- \( B \) is the predicted mask

![IoU](./image/iou.png)

### Dice Coefficient (F1-Score)

The Dice Coefficient, also known as the F1-Score, is calculated from the precision and recall of a prediction. It scores the overlap between the predicted segmentation and the ground truth. It is defined as:

\[ \text{Dice Coefficient} = \frac{2|A \cap B|}{|A| + |B|} \]

Where:
- \( A \) is the ground truth mask
- \( B \) is the predicted mask

The Dice Coefficient ranges from 0 (no overlap) to 1 (perfect overlap), with higher values indicating better performance.
 





This repository contains code for training, validating, and performing inference using a UNET model for rooftop segmentation.

## Requirements

- Python 3.x
- PyTorch
- Albumentations
- OpenCV
- Pillow
- Matplotlib
- Argparse
- Tensorboard

## Repository Structure

- `dataset.py`: Contains the `CustomDataset` class for handling dataset loading and preprocessing.
- `train.py`: Script for training the UNET model.
- `inference.py`: Script for running inference on a single image using the trained model.
- `utils.py`: Utility functions for saving/loading checkpoints and creating data loaders.

## Dataset

The dataset is expected to be organized in the following structure:
```
data_dir/
├── train/
│   ├── image1/
│   │   ├── images/
│   │   │   └── image1.jpg
│   │   ├── masks/
│   │   │   ├── mask1.png
│   │   │   └── mask2.png
│   └── image2/
│       ├── images/
│       │   └── image2.jpg
│       ├── masks/
│           ├── mask1.png
│           └── mask2.png
├── validation/
│   ├── image1/
│   │   ├── images/
│   │   │   └── image1.jpg
│   │   ├── masks/
│   │   │   ├── mask1.png
│   │   │   └── mask2.png
│   └── image2/
│       ├── images/
│       │   └── image2.jpg
│       ├── masks/
│           ├── mask1.png
│           └── mask2.png
```

## Training

To train the model, use the `train.py` script. You can configure the training parameters through command line arguments.

```bash
python train.py --num_epochs 20 --batch_size 8 --learning_rate 0.0001 --image_height 160 --image_width 240 --train_dir ./data/train --test_dir ./data/validation
```

## Inference

To run inference on a single image, use the `inference.py` script. 

```bash
python inference.py --image_path ./data/test/image1.jpg --model_path ./models/final.pth.tar
```

## Utilities

- `utils.py`:
  - `save_checkpoint(state, model_name="my_checkpoint.pth.tar")`: Saves the model checkpoint.
  - `load_checkpoint(checkpoint_path, model)`: Loads the model checkpoint.
  - `get_loaders(TRAIN_DIR, VAL_DIR, BATCH_SIZE, train_transform, val_transform, NUM_WORKERS, PIN_MEMORY=True)`: Creates training and validation data loaders.

## Example Commands

### Train the model

```bash
python train.py --num_epochs 20 --batch_size 8 --learning_rate 0.0001 --image_height 160 --image_width 240 --train_dir ./data/train --test_dir ./data/validation --load_model True --load_model_path ./models/final.pth.tar --verbose True
```

### Run inference on an image

```bash
python inference.py --image_path ./data/test/image1.jpg --model_path ./models/final.pth.tar
```

## Notes

- Ensure the directory paths for training, validation, and test data are correctly specified.
- Adjust the hyperparameters as needed for optimal performance.
- Use `tensorboard` to visualize training metrics by running `tensorboard --logdir=./logs`.

## Acknowledgements

This project uses the following libraries and frameworks:
- PyTorch
- Albumentations
- OpenCV
- Pillow
- Matplotlib

Feel free to raise issues or contribute to this repository.
```

This `README.md` file provides a clear and structured overview of the repository, including instructions on how to train the model, run inference, and understand the directory structure and utilities provided.