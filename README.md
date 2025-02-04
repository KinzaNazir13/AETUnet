README: Medical Image Segmentation using PyTorch
Overview
This project implements a medical image segmentation model using PyTorch. It includes custom data augmentation techniques, dataset management, and training scripts for efficient and robust segmentation.
Repository Structure
•	transforms.py: Contains data augmentation and preprocessing functions.
•	train.py: Main script for training and evaluating the segmentation model.
•	Segmentation_dataset.py: Custom dataset class for loading images and masks.
Setup
Requirements
•	Python 3.x
•	PyTorch
•	torchvision
•	numpy
•	PIL
•	matplotlib
Installation
Install the required Python packages:
bash
CopyEdit
pip install torch torchvision numpy pillow matplotlib
Usage
1. Data Preparation
Organize the dataset as follows:
bash
CopyEdit
Dataset/
│
└───train/
│   ├───images/
│   └───1st_manual/
└───test/
    ├───images/
    └───1st_manual/
•	images/ contains the input images (e.g., .jpg files).
•	1st_manual/ contains the ground truth masks (e.g., .tif files).
2. Training
Run the training script:
bash
CopyEdit
python train.py --data-path "path_to_dataset" --num-classes 1 --epochs 100 --batch-size 2 --lr 0.0015
Optional arguments:
•	--device: Device to use (default: "cuda").
•	--resume: Path to resume training from a checkpoint.
•	--save-best: Save only the best model based on Dice coefficient.
3. Evaluation
The script automatically evaluates the model after each epoch, displaying metrics like Dice coefficient and accuracy.
4. Visualization
The visualize() function in train.py can be used to display the original image, ground truth, and segmented output:
python
CopyEdit
visualize(image, output, label)
Custom Transforms
Custom data augmentation techniques are implemented in transforms.py:
•	RandomResize: Resize images randomly within a specified range.
•	RandomHorizontalFlip & RandomVerticalFlip: Flip images with a given probability.
•	RandomCrop & CenterCrop: Crop images randomly or from the center.
•	Normalize & ToTensor: Normalize image tensors and convert to PyTorch tensors.
Dataset
The SegmentationDataset class in Segmentation_dataset.py handles loading and preprocessing:
•	Loads images and corresponding masks.
•	Converts masks to binary format.
•	Applies specified transforms.
Model
The segmentation model is based on a custom UNet architecture (AETUnet), supporting:
•	Adjustable input channels.
•	Adjustable number of output classes.
Results
The results, including loss and accuracy, are logged in a text file with a timestamp.
Checkpoints
Model checkpoints are saved in the save_weights/ directory. By default, the best-performing model (based on Dice coefficient) is saved.
