
# CS4486 Iceberg Classification Mini Project

## Project Overview

This project addresses the Statoil/C-CORE Iceberg Classifier Challenge using SAR images and incidence-angle metadata.

Our final approach combines an ImageNet-pretrained EfficientNet-B0 backbone, multimodal feature fusion, five-fold stratified cross-validation, and SMAC hyperparameter optimization.

## Dataset

Kaggle Competition:
https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge

The dataset includes 1,604 labeled training samples and 8,424 unlabeled test samples.

## Code Files

- `iceberg-experiments.ipynb`: Model development, augmentation experiments, hyperparameter optimization, and ablation studies.
- `iceberg-final-submission.ipynb`: Final five-fold model training, test prediction, and submission CSV generation.

## Final Configuration

- Backbone: EfficientNet-B0 (ImageNet pretrained)
- Input: SAR image and incidence angle
- Augmentation: Random 90-degree rotations
- Optimizer: Adam
- Learning rate: 0.00135413
- Weight decay: 0
- Dropout: 0.09333
- Label smoothing: 0.00339
- Learning rate scheduler: Cosine annealing
- Early stopping: Patience 8
- Validation: Stratified 5-fold CV

## Results

- Mean 5-fold CV Log Loss: 0.187174
- Kaggle Public Log Loss: 0.16161
- Kaggle Private Log Loss: 0.15291
- Kaggle Private Rank: 417

## Reproduction

1. Open the notebooks in Kaggle.
2. Attach the official competition dataset.
3. Use a GPU runtime and install the required dependencies.
4. Run the experiment notebook to reproduce the development process.
5. Run the final submission notebook to generate `submission.csv`.

## Environment

See `requirements.txt` for the software dependencies and versions.
