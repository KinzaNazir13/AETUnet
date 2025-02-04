# AETUnet: Lightweight UNet for Retinal Lesion Segmentation  

## Overview  
AETUnet is a lightweight and efficient deep learning model based on the UNet architecture, designed for retinal lesion segmentation. It integrates **large-kernel depth-wise separable convolutions** and a **lightweight attention mechanism** to improve segmentation accuracy while minimizing computational overhead.  

## Features  
- **Efficient Segmentation**: Optimized for medical image analysis with lower computational cost.  
- **Lightweight Design**: Uses depth-wise separable convolutions to reduce parameters.  
- **Attention Mechanism**: Enhances feature refinement while maintaining efficiency.  
- **Data Augmentation**: Supports resizing, flipping, cropping, and normalization.  

---

## Installation  

### **1. Clone the Repository**  

git clone https://github.com/KinzaNazir13/AETUnet.git
cd AETUnet
## **2. Install Dependencies**
Ensure you have Python 3.7+ and install the required packages:

pip install torch torchvision numpy pillow matplotlib

## **Dataset Structure**
Place your dataset inside a folder and follow this structure:


     ```bash
         Dataset/
         │── train/
         │   ├── images/         # Input images
         │   ├── 1st_manual/     # Ground truth segmentation masks
         │── test/
         │   ├── images/  
         │   ├── 1st_manual/  
Modify the dataset path in train.py (default: D:\HardExudates). 

## **Training the Model**
Run the following command to start training:
python train.py --data-path /path/to/dataset --epochs 100 --batch-size 4 --lr 0.0015

     ```bash
           Training Parameters:
           --data-path → Path to dataset folder
           --epochs → Number of training epochs (default: 100)
           --batch-size → Number of samples per batch (default: 4)
           --lr → Learning rate (default: 0.0015)

## Model
The segmentation model is based on a custom UNet architecture (AETUnet), supporting:

Adjustable input channels.
Adjustable number of output classes.
## Results
The results, including loss and accuracy, are logged in a text file with a timestamp.

## Acknowledgments
This repository is part of the research work on lightweight deep learning architectures for medical image segmentation. If you use this code, please consider citing the corresponding paper.
