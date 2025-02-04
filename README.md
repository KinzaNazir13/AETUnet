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
```bash
git clone https://github.com/KinzaNazir13/AETUnet.git
cd AETUnet



Here is the README.md file in a copy-paste format. You can simply copy this and paste it directly into your GitHub repository as README.md.

markdown
Copy
Edit
# AETUnet: Lightweight UNet for Retinal Lesion Segmentation  

## Overview  
AETUnet is a lightweight and efficient deep learning model based on the UNet architecture, designed for retinal lesion segmentation. It integrates **large-kernel depth-wise separable convolutions** and a **lightweight attention mechanism** to improve segmentation accuracy while minimizing computational overhead.  

## Features  
- **Efficient Segmentation**: Optimized for medical image analysis with lower computational cost.  
- **Lightweight Design**: Uses depth-wise separable convolutions to reduce parameters.  
- **Attention Mechanism**: Enhances feature refinement while maintaining efficiency.  
- **Data Augmentation**: Supports resizing, flipping, cropping, and normalization.  

---


Here is the README.md file in a copy-paste format. You can simply copy this and paste it directly into your GitHub repository as README.md.

markdown
Copy
Edit
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
```bash
git clone https://github.com/KinzaNazir13/AETUnet.git
cd AETUnet
2. Install Dependencies
Ensure you have Python 3.7+ and install the required packages:

pip install torch torchvision numpy pillow matplotlib

Dataset Structure
Place your dataset inside a folder and follow this structure:
Dataset/
│── train/
│   ├── images/         # Input images
│   ├── 1st_manual/     # Ground truth segmentation masks
│── test/
│   ├── images/  
│   ├── 1st_manual/  
Modify the dataset path in train.py (default: D:\HardExudates).

Training the Model
Run the following command to start training:
python train.py --data-path /path/to/dataset --epochs 100 --batch-size 4 --lr 0.0015

