# Skin Disease Image Classification using MobileNetV2

A deep-learning image classification project that uses **MobileNetV2 transfer learning** to classify skin images into **10 skin-condition categories**. The model was trained and evaluated on a dataset of 27,153 images and achieved a reported **test accuracy of 72.23%**.

> **Important medical disclaimer:** This is an educational machine-learning project, not a clinically validated medical device. It must not be used to diagnose skin conditions, choose treatments, or replace advice from a qualified healthcare professional.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Technology Stack](#technology-stack)
- [Model Architecture](#model-architecture)
- [Training Workflow](#training-workflow)
- [Results](#results)
- [How to Run](#how-to-run)
- [Expected Dataset Layout](#expected-dataset-layout)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Disclaimer](#disclaimer)

## Project Overview

Skin conditions can look similar in photographs, making image classification a useful machine-learning task to explore. This project builds an image classifier using **TensorFlow/Keras** and **MobileNetV2**, a convolutional neural network pretrained on ImageNet.

The pretrained feature extractor is adapted to the project's 10 output classes. The training process has two main phases:

1. **Transfer learning:** Freeze the pretrained MobileNetV2 base and train a new classification head.
2. **Fine-tuning:** Unfreeze the final 30 backbone layers for additional training with a lower learning rate, while keeping Batch Normalization layers frozen.

The model produces a probability distribution across the 10 dataset classes. Its highest-probability class is the predicted category; the prediction should not be interpreted as a clinical diagnosis.

## Objectives

- Prepare and process skin-image data for model training.
- Use transfer learning to reduce the amount of training required from scratch.
- Apply data augmentation to improve generalization.
- Address class imbalance using class weights.
- Evaluate the model on a held-out test set using accuracy and F1-score metrics.
- Identify classes where the model performs well and where further improvements are needed.

## Dataset

The experiment used **27,153 usable images** across 10 categories. The project report states that zero empty files were removed.

| Class | Number of images |
|---|---:|
| Eczema | 1,677 |
| Warts, Molluscum and Other Viral Infections | 2,103 |
| Melanoma | 3,140 |
| Atopic Dermatitis | 1,257 |
| Basal Cell Carcinoma (BCC) | 3,323 |
| Melanocytic Nevi (NV) | 7,970 |
| Benign Keratosis-like Lesions (BKL) | 2,079 |
| Psoriasis, Lichen Planus and Related Diseases | 2,055 |
| Seborrheic Keratoses and Other Benign Tumors | 1,847 |
| Tinea/Ringworm, Candidiasis and Other Fungal Infections | 1,702 |
| **Total** | **27,153** |

### Dataset split

The images were divided using a **stratified 70% / 15% / 15% split**:

| Split | Images | Approx. proportion |
|---|---:|---:|
| Training | 19,007 | 70% |
| Validation | 4,073 | 15% |
| Test | 4,073 | 15% |
| **Total** | **27,153** | **100%** |

Stratification helps preserve the relative class distribution across the splits. The test set is used for final evaluation rather than routine model-selection decisions.

## Technology Stack

- **Language:** Python 3.10.11
- **Deep learning:** TensorFlow 2.10.0 / Keras
- **Model:** MobileNetV2 pretrained on ImageNet
- **Data pipeline:** `tf.data`
- **Image processing:** TensorFlow image decoding and resizing
- **Evaluation:** Accuracy, precision/recall-derived F1 scores, and classification report metrics
- **Runtime noted in the project report:** Windows 10, with GPU device checks in the TensorFlow environment

GPU availability and setup depend on the operating system, TensorFlow build, and compatible drivers. Seeing a GPU in a device check does not by itself prove that every operation or the complete training process ran on the GPU.

## Model Architecture

The network uses the following components:

1. **Input:** RGB images resized to `224 × 224` pixels.
2. **Preprocessing:** MobileNetV2's `preprocess_input` function.
3. **Data augmentation:** Random horizontal flip, random rotation (`0.05`), and random zoom (`0.10`).
4. **Base model:** MobileNetV2 pretrained on ImageNet with `include_top=False` to remove its original classification head.
5. **Pooling:** `GlobalAveragePooling2D` to convert feature maps into a feature vector.
6. **Regularization:** `Dropout(0.35)` to reduce overfitting.
7. **Output:** A dense layer with 10 units and `softmax` activation, returning a probability for each class.

The initial model summary reported **2,270,794 total parameters**, of which **12,810 were trainable** and **2,257,984 were non-trainable** during the first phase.

## Training Workflow

### 1. Image input pipeline

- Decode images as three-channel RGB.
- Resize each image to `224 × 224`.
- Convert image values to `float32`.
- Create batches of **16** images.
- Use prefetching to help prepare batches efficiently.
- Apply MobileNetV2 preprocessing within the model.

### 2. Transfer learning

The pretrained MobileNetV2 backbone is initially frozen. The new classification head is trained using:

- **Loss:** Sparse categorical cross-entropy
- **Optimizer:** Adam
- **Learning rate:** `0.001`
- **Batch size:** `16`

### 3. Fine-tuning

For the second phase, the final **30 backbone layers** are made trainable, while Batch Normalization layers remain frozen. The learning rate is reduced to `0.00001` to make smaller updates to the pretrained weights.

### 4. Imbalance and training controls

- **Class weights** are used to account for differences in the number of images per class.
- **EarlyStopping** restores the best weights when validation progress stalls.
- **ReduceLROnPlateau** lowers the learning rate when the monitored validation metric stops improving.
- **ModelCheckpoint** saves a model checkpoint during training.

## Results

The following results are from the project report.

### Overall performance

| Metric | Initial transfer-learning phase | Fine-tuned model |
|---|---:|---:|
| Validation accuracy | 66.22% | 71.23% |
| Validation loss | 0.8878 | 0.7556 |

The fine-tuned model was selected because it had higher validation accuracy and lower validation loss.

| Final test metric | Result |
|---|---:|
| Test accuracy | **72.23%** |
| Test loss | 0.7281 |
| Macro F1-score | 0.650 |
| Weighted F1-score | Approximately 0.723 |

### Selected class-level F1-scores

| Class | F1-score |
|---|---:|
| Melanocytic Nevi (NV) | 0.876 |
| Melanoma | 0.851 |
| Basal Cell Carcinoma (BCC) | 0.821 |
| Eczema | 0.466 |
| Atopic Dermatitis | 0.495 |
| Psoriasis, Lichen Planus and Related Diseases | 0.495 |

The scores show that performance differs by class. In particular, the weaker scores for eczema, atopic dermatitis, and the psoriasis/lichen-planus-related group indicate that the model still confuses some categories. Overall accuracy alone does not describe the risk of every kind of error, especially for medical images.

## How to Run

The project report documents the environment and workflow but does not establish a single universal installation command or the exact notebook/script filename. Use the notebook or Python entry point in your own project repository and adapt the dataset path to your environment.

### 1. Prepare the environment

Use a Python environment compatible with the TensorFlow version used by the project. The recorded version was TensorFlow `2.10.0` with Python `3.10.11`.

Install the packages required by your notebook/script. At minimum, the core deep-learning dependency is:

```bash
pip install tensorflow==2.10.0
```

Install any additional packages imported by your code. If you intend to use a GPU, check TensorFlow and GPU/driver compatibility for your operating system before installing a GPU-enabled setup; do not assume that the command above enables GPU acceleration on every machine.

### 2. Arrange the dataset

Place the dataset in a directory containing one folder per class, with the corresponding images inside each folder. See [Expected Dataset Layout](#expected-dataset-layout).

### 3. Configure the dataset path

Update the dataset path in the notebook/script to point to the parent directory containing the 10 class folders. Make sure that the class folders and image files match the labels expected by the code.

### 4. Run training and evaluation

Run the notebook or script in the following order:

1. Load and inspect image paths and class names.
2. Create the stratified training, validation, and test splits.
3. Build the image input pipeline and model.
4. Train the classification head.
5. Fine-tune the final backbone layers.
6. Evaluate the selected model on the held-out test set.
7. Review overall metrics and the per-class classification report.

**Reproducibility note:** Results can vary with random seeds, data splitting, package versions, hardware, and training settings. The metrics above are reported experimental results, not a guarantee that another run will reproduce exactly the same values.

## Expected Dataset Layout

The following is an example layout for the data loader. Folder names should match the actual labels in your dataset and code.

```text
project-folder/
├── README.md
├── notebook-or-training-script
└── skin/
    ├── Eczema/
    │   ├── image_001.jpg
    │   └── ...
    ├── Melanoma/
    │   ├── image_001.jpg
    │   └── ...
    ├── Atopic Dermatitis/
    │   └── ...
    └── ... (remaining class folders)
```

This is an illustrative layout, not a verified listing of the original repository. Do not commit the full image dataset to GitHub unless you have the rights to redistribute it and it complies with the dataset's license and privacy terms.

## Limitations and Future Improvements

Potential next steps include:

- Inspecting a confusion matrix to identify which classes are frequently confused.
- Reviewing precision, recall, F1-score, and per-class sample counts together, rather than relying only on accuracy.
- Improving and validating class-specific performance, especially for the lower-scoring classes.
- Checking for duplicate images, label quality issues, and possible data leakage between splits.
- Evaluating robustness across image sources, lighting, skin tones, and acquisition devices.
- Performing external validation on a separately collected and appropriately licensed dataset.
- Calibrating model probabilities and documenting uncertainty before considering any real-world use.
- Seeking expert clinical review and relevant regulatory/ethical assessment before any medical deployment.

These are recommendations for future work; they are not claims that the checks or improvements have already been completed.

## Disclaimer

This repository is intended for educational and research purposes only. The model's output is a statistical prediction from the dataset classes and may be incorrect. It has not been established as safe or effective for clinical use. **Do not use this project to diagnose, rule out, or treat a skin condition.** Consult a qualified healthcare professional for medical concerns.
