# A fast Nearest Neighbor Scarch Scheme over Outsourced Encrypted Medical Images

DATASET=https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Pneumonia Detection from Chest X-Rays (Label Correction and Model Training)

This project aims to build a robust AI system to detect pneumonia from chest X-ray images, using the “Chest X-Ray Images (Pneumonia)” dataset by Paul Timothy Mooney on Kaggle. 
Kaggle

A key focus of this work is correcting mislabelled data, especially where images labelled “normal” were in fact pneumonia (or vice versa), to improve the model’s accuracy and reliability.

Key Components:

Data Cleaning & Label Verification

Manually review ambiguous or edge-case X-ray images.

Use visualization, metadata, and domain heuristics to correct misclassifications.

Ensure the “normal” vs “pneumonia” labels accurately reflect the underlying condition.

Preprocessing

Resize images to consistent dimensions.

Normalize pixel values.

Augment data (rotations, flips, brightness/contrast adjustments) to improve model generalization.

Model Architecture & Training

Use CNN architectures (e.g. ResNet, DenseNet, Inception) fine-tuned on pre-trained weights.

Train with cross-validation.

Monitor metrics including accuracy, precision, recall, F1-score, and ROC-AUC to ensure balanced performance, particularly on minority class (if classes are imbalanced).

Evaluation & Fixes

Check confusion matrix: watch especially false negatives (cases labelled “normal” but actually pneumonia).

Use techniques like Grad-CAM or other explainability methods to verify what the model is focusing on.

Possibly use ensemble methods or threshold adjustment to reduce harmful misclassification.

Deployment / User Interface (if applicable)

A simple web interface or notebook-based demo where users can upload chest X-ray images and receive a “normal / pneumonia” classification.

Include visual explanations (e.g., heatmap overlay) if possible, to make diagnosis more interpretable.
