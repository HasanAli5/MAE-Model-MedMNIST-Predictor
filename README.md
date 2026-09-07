# MAE-Model-MedMNIST-Predictor
Using Masked Autoencoder model pretraining to Predict Medical Imaging Results.

# For Weights of the Models:
[Hugging Face Model Weights](https://huggingface.co/Hali5/Mae-Model-MedMNIST-Predictor)

## For Live Demo:
[Hugging Face Model Space Demo](https://huggingface.co/spaces/Hali5/MAE-PathMNIST-Predictor-Demo)

## Results:

The best predictor variant was the cross attention pooler with 95.11% test accuracy.

This test accuracy beats the best ResNet (ResNet-50 (28)) model accuracy from the official MedMNIST v2 dataset benchmark by 4%.

Here are the variants and their test accuracies and AUC:

|         Variant          |      Accuracy      |         AUC       |
|--------------------------|--------------------|-------------------|
|  Benchmark  ResNet-50 (28)    |  0.911 | 0.990 |
|    Linear Probe (GAP)    | 0.9415 | 0.9942 |
| Cross Attention Pooler   | **0.9511** | 0.9942 |
| Cross Attention Hybrid   | 0.9433 | **0.9954** |
