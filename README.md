# Slash_Image_Classification
To achieve this taks, you will be given a data set that consists of 427  images to train your model and to tune your hyperparameters. However, feel free to extend it by collecting new images or by using data augmentation techniques.
Setup the environment: Thie first step consists of setting the environement and downloading the data.
Preprocessing: The second step is a preprocessing step that consists of resizing, plitting, and piping the input data.
Exploring the data: The third step consists of a simple data exploration step where you will see samples of the data and some statistics to help you in understanding the data.
Designing the model: The forth step consists of designing an architecture for the task.
Traning: The fifth step consists of starting the training process.
Monitoring: The sixth step consists of monitoring the traning process to investigate possible overfitting.
Compiling the model by defininf an optimizer, a loss function, and the metrics to be used for monitoring the traning ===> model using Stochastic Gradient Descent optimizer with a learning rate of 0.01 and momentum of 0.7, After applying Parameter tuning ,Categorical Crossentropy as the loss function, and accuracy as the evaluation metric.
conclusion about the metric:The overall accuracy of the model is 0.64, which means it correctly predicts the class labels for 64% of the samples. The macro average F1-score (0.33) and weighted average F1-score (0.55) reflect the overall performance across all classes, showing room for improvement, particularly in classes with low precision, recall, and F1-score.
