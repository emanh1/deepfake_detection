# Deepfake Detection

This project implements a deepfake detection pipeline using PyTorch and computer vision techniques. The workflow includes dataset preparation, frame and face extraction, model training (from scratch and with pretrained networks), hyperparameter tuning, and evaluation.

## Features
- **Dataset Download & Preparation**: Automatically downloads and unpacks the Deep Fake Detection (DFD) dataset.
- **Frame Extraction**: Extracts frames from real and fake videos.
- **Face Extraction**: Detects and crops faces from frames using MTCNN.
- **Data Augmentation**: Applies various transformations for robust training.
- **Model Architectures**:
  - Custom ResNet18 (from scratch)
  - Pretrained ResNet18 (transfer learning)
- **Training & Evaluation**: Includes training loops, validation, and plotting of metrics.
- **Hyperparameter Tuning**: Uses Optuna for automated hyperparameter search.
- **Visualization**: Plots training/validation accuracy, loss, and confusion matrices.

## Usage
1. **Install Requirements**
   - Python 3.8+
   - PyTorch, torchvision, scikit-learn, matplotlib, tqdm, opencv-python, mtcnn, optuna, seaborn, pillow
   - Install with:  
     `pip install torch torchvision scikit-learn matplotlib tqdm opencv-python mtcnn optuna seaborn pillow`
2. **Run the Notebook**
   - Open `deepfake_detection.ipynb` in VS Code or Jupyter.
   - Execute cells sequentially to download data, extract frames/faces, train models, and evaluate results.

## Files
- `deepfake_detection.ipynb`: Main notebook with all code and explanations.
- `pretrained_model_best.pth`, `pretrained_model_final.pth`: Saved weights for pretrained model.
- `scratch_model_best.pth`, `scratch_model_final.pth`: Saved weights for scratch model.
- `plot_*.png`: Training/validation plots.