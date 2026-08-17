# MAE-Model-MedMNIST-Predictor
Using Mixed Autoencoder Model Architecture to Predict Medical Imaging Results.

# For Weights of the Models:
[Hugging Face Repo Containing Weights](https://huggingface.co/Hali5/Mae-Model-MedMNIST-Predictor)

## For Live Demo:
[Hugging Face Space Demo](https://huggingface.co/spaces/Hali5/MAE-PathMNIST-Predictor-Demo)

## Results:

The best predictor variant was the linear probe with 91.448% test accuracy, check the logs for the test accuracies of the other predictor variants.

This test accuracy beats the best ResNet (ResNet-50 (28)) model accuracy from the official MedMNIST v2 dataset benchmark by around 0.3%.

Here are the variants and there test accuracy and AUC:

|         Variant          |      Accuracy      |         AUC       |
|--------------------------|--------------------|-------------------|
|    Linear Probe (GAP)    | 0.9145 | 0.9887 |
| Cross Attention Pooler   | 0.8947 | 0.9863 |
| Cross Attention Ones     | 0.8978 | 0.9846 |
| Cross Attention Identity | 0.9007 | 0.9853 |
| Cross Attention Hybrid   | 0.9075 | 0.9821 |


Notably for the cross attention pooling methods, the convergence time is significantly reduced where it would only require the first 20 to 30 epochs (of 100) to be able to get the loss plateau, and where no significant improvement to accuracy happens afterwards. However the final test accuracy is lower than the linear probe using global average pooling (GAP) where methods like the cross attention hybrid might be viable if you are looking for best performance in the quickest time to train.

## To Do

* make a second hybrid variant that mixes cross attention identity and Linear Probe (GAP)
