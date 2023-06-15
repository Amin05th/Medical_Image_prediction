# Medical_Image_prediction

- Created a AI that classifies if a person has cancer or not to help doctors with the analysation.
- Downloaded dataset from Kaggle and extrected dataset into my folder.
- Transformed and normilized images using torchvision.
- Preprocessed data with and downloaded data with dataset module from pytorch and dataloader.
- Tuned and optimized neural network and created test and train function

## Code and Resources Used
- **Python Version:** 3.10
- **Packages:** pytorch, cv2, glob, fnmatch, tensorboard, sklearn, skorch, numpy
- **Kaggle Dataset:** [https://www.kaggle.com/c/nlp-getting-started/overview](https://www.kaggle.com/c/nlp-getting-started/overview)

## Neural Network building
First I preprocessed Images using transforms then I split data into train, test and validation set with validation size of 20%.

I tried a neural network with 3 Sequential layer (with Conv, BatchNorm2, LeakyReLU, MaxPool2d) and with 4 fully connected layer
As a Loss function I used cross entropy, as optimizer I used Adam and as a scheduler I used StepLR

## Hyperparameter tuning
For hyperparameter tuning I used skorch. I tuned all 3 hidden layer, learn rate and epochs and I took the best model possible.
