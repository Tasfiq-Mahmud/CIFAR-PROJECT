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
- **Lower validation accuracy**
- **Higher instability in loss behavior**

indicating that the ensemble dropout setup did **not** improve generalization for this configuration.

Observations
- The base model was already well-regularized
- Introducing stochasticity at inference added noise rather than robustness
- The small input resolution of CIFAR-10 likely limits the benefits of Monte-Carlo Dropout

Below is a visualization of the performance trends observed during this experiment:

<img width="1500" height="500" alt="Ensemble Dropout Experiment Results" src="https://github.com/user-attachments/assets/7be33a7f-914f-407b-b991-81506ad21879" />

This experiment reinforces the importance of **empirical validation over theoretical assumptions**.
