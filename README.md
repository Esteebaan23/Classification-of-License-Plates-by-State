# License Plate State Recognition 🚗🗽

The purpose of this project is to develop the license plate classification phase within an ALPR system focused on identifying license plates from different U.S. states. From the user's perspective, it seeks to facilitate the work of road safety agencies, police departments and operators of intelligent transportation systems. By enabling accurate classification of license plates by state, the system can help track vehicles involved in crime, toll evasion, theft or interstate pursuits, with the hope that the project will contribute to a safer, smarter and more connected road environment.

This project focuses on developing a deep learning system for recognizing U.S. license plate states using image classification techniques. Multiple CNN architectures and Transformer-based models were benchmarked, and an ensemble method was implemented to maximize classification accuracy.

## Dataset
- **Source**: [Kaggle - U.S. License Plates Dataset]([https://www.kaggle.com/datasets/gpiosenka/us-license-plates-image-classification](https://www.kaggle.com/datasets/gpiosenka/us-license-plates-image-classification))
- **Classes**: 50 U.S. states
- **Preprocessing**:
  - Data augmentation (Random Rotation, Flip, Zoom)

## Models
The following architectures were trained and evaluated:
1. **ResNet50** (CNN.py)
2. **DenseNet121** (CNN.py)
3. **VGG16** (CNN.py)
4. **Vision Transformer (ViT Base - DeiT3)**  (ViT.py)
5. **Custom CNN with DenseBlocks & Transition Layers** (Custom_CNN.py)

Additionally, a **Stacking Ensemble Model** was implemented, combining the predictions of ResNet50, DenseNet121, and ViT to improve overall performance.

## Results
| Model                        | Accuracy (%) | Notes                                        |
|------------------------------|--------------|----------------------------------------------|
| ResNet50                     | 92.47        | Best single model performance                |
| DenseNet121                  | 91.82        | Slightly lower performance than ResNet50     |
| VGG16                        | 89.34        | Underperformed compared to newer architectures |
| Vision Transformer (ViT DeiT3)| 91.25        | High accuracy with fewer parameters          |
| Custom CNN                   | 90.71        | Lightweight but slightly less accurate       |
| **Stacking Ensemble**        | **93.52**    | Best overall accuracy by combining models    |


## Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/Esteebaan23/Classification-of-License-Plates-by-State.git
    cd Classification-of-License-Plates-by-State
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run inference on a sample image:
    ```bash
    python main.py 
    ```

4. Launch the Streamlit App:
    ```bash
    streamlit run main.py
    ```

