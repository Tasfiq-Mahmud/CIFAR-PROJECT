# CIFAR-10 Image Classification — Learning-Focused Deep Learning Project

## Overview
This project explores **image classification on the CIFAR-10 dataset using PyTorch**, with an emphasis on **understanding training dynamics rather than chasing benchmark scores**.

The goal was to build strong fundamentals in:
- model regularization
- data augmentation
- optimization
- generalization vs overfitting

Every technique used here was **tested, questioned, and validated experimentally**.

---

## What I Learned Through This Project

- How regularization techniques (Dropout, Weight Decay) affect generalization
- Why some popular augmentations **hurt performance on small images**
- How learning rate schedulers influence convergence
- How normalization actually works (not just how to apply it)
- How architectural changes affect model saving/loading
- Why validation accuracy can increase while loss behaves differently

---

## Model Architecture

- Backbone: **ResNet-18**
- Modifications:
  - Final classification layer adapted for CIFAR-10 (10 classes)
  - **Dropout layer added before the classifier** to improve generalization and enable ensemble-style inference

## Data Augmentations Used

- Dropout during training -> Probability = 0.2
- RandomCrop(size=32,padding=4)
- RandomHorizontalFlip(p=0.5)
- Normalization

```text
ResNet-18
 └── Feature extractor (unchanged)
 └── Dropout(p)
 └── Linear(512 → 10)
```

## Observations on Data Augmentation

During experimentation, **RandomCrop with padding consistently outperformed RandomResizedCrop** on the CIFAR-10 dataset. In addition to achieving better validation accuracy, RandomResizedCrop also **significantly slowed down training iterations**.

### RandomCrop with Padding
- Preserves original image resolution
- Introduces translation invariance without resizing
- Maintains fine-grained texture information

Below is a visualization and performance trend when using **RandomCrop(32, padding=4)**:

<img width="1500" height="500" alt="RandomCrop Augmentation Results" src="https://github.com/user-attachments/assets/a7fb04af-e72b-4431-a79e-01609d58f49b" />

---

### RandomResizedCrop
- Randomly crops a smaller region and resizes it back to the original size
- Introduces interpolation blur on 32×32 images
- Increases computational overhead due to resampling
- Led to degraded validation performance in this setup

Below is a visualization and performance trend when using **RandomResizedCrop**:

<img width="1500" height="500" alt="RandomResizedCrop Augmentation Results" src="https://github.com/user-attachments/assets/7f554532-067d-4eef-b66f-d0c6136d1e53" />

---

### Key Takeaway
For **small-resolution datasets like CIFAR-10**, augmentations that involve **resizing** can be destructive.  
Simple spatial augmentations such as **RandomCrop with padding** provide effective regularization without sacrificing image fidelity.

These results highlight the importance of **choosing augmentations based on dataset characteristics rather than default best practices**.

## Dropout Ensemble Experiment (Negative Result)

An additional experiment was conducted using an **ensemble-style inference approach with Dropout (p = 0.2)** enabled during evaluation.

Contrary to expectations, this approach resulted in:
- **Slightly Lower validation accuracy**
- **But Drastic Lower Test Accuracy. (87%->83%)**

indicating that the ensemble dropout setup did **not** improve generalization for this configuration.

Observations
- The base model was already well-regularized
- Introducing stochasticity at inference added noise rather than robustness
- The small input resolution of CIFAR-10 likely limits the benefits of Monte-Carlo Dropout

Below is a visualization of the performance trends observed during this experiment:

<img width="1233" height="468" alt="image" src="https://github.com/user-attachments/assets/9ce8fbc6-86a9-45d8-8ebf-935d88d021c0" />


This experiment reinforces the importance of **empirical validation over theoretical assumptions**.


## ColorJitter Experiment

Color-based augmentation using **ColorJitter** was evaluated to test whether photometric variations would improve model generalization.

### Observations
- Applying ColorJitter resulted in **little to no improvement in validation accuracy**
- In some runs, training became slower without measurable gains
- Aggressive jitter settings occasionally degraded performance

### Interpretation
CIFAR-10 images are:
- low resolution (32×32)
- already noisy
- not strongly color-dependent for many classes

As a result, introducing heavy color perturbations did **not add meaningful invariance** and sometimes introduced unnecessary noise.

### Takeaway
Color-based augmentations are **not universally beneficial**.  
For CIFAR-10, **mild or no ColorJitter** performed better than aggressive color transformations.

This further emphasizes the need to **match augmentation strategies to dataset characteristics**, rather than applying them blindly.


## EfficientNet-B0 vs ResNet: Validation–Test Discrepancy Analysis

After establishing a baseline using a ResNet architecture, the model backbone was replaced with EfficientNet-B0 to evaluate its impact on classification performance on CIFAR-10.

Validation Performance

The EfficientNet-B0 model achieved a validation accuracy of approximately 96%, which was significantly higher than the training accuracy. While this initially appeared promising, the unusually high validation score raised concerns regarding potential overfitting to the validation distribution or sensitivity to augmentation and data splits.

<p align="center"> <img src="https://github.com/user-attachments/assets/703986a9-36c8-44db-921f-cdac3dfcd8ad" width="900"> </p>
Test Performance and Generalization Gap

Despite the strong validation results, the test accuracy reached only 88.4%. Although this represents an improvement of approximately 1% over the ResNet baseline, the large gap between validation and test performance contrasts with earlier ResNet experiments, where validation and test accuracies were closely aligned.

This discrepancy suggests that EfficientNet-B0, due to its higher parameter efficiency and representational capacity, may be:

- overfitting to validation-specific patterns,
- exploiting augmentation-induced artifacts,
- or benefiting from an optimistic validation split that does not fully reflect the test distribution.

## Augmentation Sensitivity Analysis

To further investigate the generalization issue, misclassified test samples were analyzed, and additional data augmentation strategies were explored. In particular, random rotation was introduced to increase invariance.

However, this modification led to a reduction in validation accuracy to approximately 93–94%, accompanied by a ~1% decrease in test accuracy.

<p align="center"> <img src="https://github.com/user-attachments/assets/3d8a5f7a-0292-4741-b098-c4f64086daf6" width="900"> </p>

This result indicates that random rotation may be detrimental for CIFAR-10, where object orientation often carries semantic meaning (e.g., vehicles and animals). Given the low resolution (32×32) of the dataset, aggressive geometric transformations can distort class-discriminative features rather than improve robustness.
