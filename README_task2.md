# Face Alignment with a CNN

A convolutional neural network that predicts 5 facial landmarks (eyes, nose, mouth corners) on face images, trained and evaluated with a focus on robustness to real-world distortions.

## Problem

Predict the (x, y) pixel coordinates of 5 facial landmarks per image. The dataset includes significant variation in head pose, lighting, image quality, and occlusion, so a model needs to generalise beyond clean, front-facing portraits.

## Approach

**Baseline.** Before building any model, I computed the simplest possible baseline: always predicting the mean landmark positions from the training set. This sets a numerical floor that any real model needs to clear.

**Model.** A compact CNN (three convolutional blocks, a dense layer, dropout, and a 10-unit linear regression head) trained from scratch to predict all 5 landmark coordinates jointly. Images are resized down for training speed, and predictions are scaled back to original resolution before evaluation, so all reported numbers are in true pixel space.

**Evaluation.** Beyond a single accuracy number, I used a cumulative error distribution to see the full spread of performance across the validation set, and looked directly at the best and worst predictions to understand failure patterns.

**Robustness analysis.** I stress-tested the trained model against increasing levels of Gaussian noise, brightness shift, and blur, to see where and how it breaks down rather than assuming it would behave the same way on distorted inputs as on clean ones.

## Repository contents

- `Task2.ipynb`: full pipeline, from data loading through preprocessing, model training, evaluation, robustness testing, and failure analysis.

Note: the original datasets and test-prediction outputs are not included in this repository.

## Techniques used

- Convolutional neural networks for direct coordinate regression (TensorFlow/Keras)
- Baseline-first evaluation methodology
- Cumulative error distribution analysis
- Systematic robustness testing under synthetic distortions
- Qualitative failure case analysis

## Notes

This was built as a learning project. AI assistance (Claude, Anthropic) was used for parts of the pipeline design discussion and code implementation; all experiments, results, and analysis were run and verified independently.
