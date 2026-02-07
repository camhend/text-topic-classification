# Text Topic Classification 

------------------------------------------------------------------------

## Overview

The task for this dataset is to predict the topic of a document from the
term-frequency vectors. The imbalanced dataset follows a heavy tail
distribution with the dominant class having 42,647 datapoints and the
rarest class having only 198 datapoints in the training dataset.

Four models were tested:

-   K-Nearest Neighbors (KNN)
-   Linear SVC (SVC)
-   Logistic Regression (LR)
-   Complement Naive Bayes (CNB)

Random search with 100 iterations was performed on each model, including
both model-specific and model-agnostic hyperparameters.

------------------------------------------------------------------------

## Model Performance

### Figure 1: Model Validation Accuracy and Time Efficiency

4 models were tested: K-nearest neighbors (KNN), Linear SVC (SVC), Logistic Regression (Log Reg), 
and Complement Naive Bayes (CNB). To start, random search was performed with 100 iterations on each of the models, 
which included hyperparameters that were both model specific and model agnostic. Fig 1 and 2 report the highest validation accuracy for these searches.

<img width="544" height="299" alt="image" src="https://github.com/user-attachments/assets/d50a8f96-a4bc-43c6-8a0d-49c6eea399f3"/>\
Fig 1: Model validation accuracy and time efficiency.


<img width="546" height="303" alt="image" src="https://github.com/user-attachments/assets/51e24f45-7df1-40c9-86c9-c6e2cb516699" />\
Fig 2: Effect of feature transformations on validation accuracy. \
Columns left to right: term frequency–inverse document frequency, \
maximum absolute value scaling, and no scaling.

LR performed the best with decent time efficiency. Notably KNN had the worst time efficiency given the large number of neighbors calculations, 
and CNB was by far the quickest (Fig 1). For feature transformations, Fig 2 shows comparisons of no scaling to a max absolute value scaler, 
and tf-idf which is commonly used in text feature transformations. (Fig. 2)  Tf-idf consistently improved performance over the other strategies. 
Logistic Regression with tf-idf was chosen for future tuning. 

------------------------------------------------------------------------

## Feature Selection and Regularization

<img width="582" height="461" alt="image" src="https://github.com/user-attachments/assets/c7960b35-36ee-457e-a930-088538c3d043" />\
Fig 3: L1/L2 regularization effect on dev accuracy. \
L2 best accuracy=0.524, L1 best accuracy=0.519.

<img width="607" height="461" alt="image" src="https://github.com/user-attachments/assets/44658b16-16f7-4686-9cf5-34dd940ea7bc" />\
Fig 4: Effect of selecting features based on chi-squared test. The effect is reversed based on the presence of regularization. 

L1 and L2 regularization were very important for performance, with L2 regularization being slightly more performant (Fig 3). 
Interestingly, when considering feature selection, the usefulness depends on the presence of regularization (Fig 4). 
If L2 regularization is performed, dimensionality reduction actually has a negative effect on performance, 
while if regularization is absent, dimensionality reduction has the expected result of substantially improving accuracy. 
Perhaps regularization performs a sort of dimensionality reduction, making feature selection redundant. 


------------------------------------------------------------------------

## Class Imbalance

Since there is a significant imbalance in the classes, all models were susceptible to defaulting to picking the most common class. 
However, they were able to reach accuracies above a baseline of merely picking the dominant class 100% of the time, 
which would result in an accuracy of 0.372. Common strategies to deal with class imbalance attempt to equalize all classes either 
through balanced class weighting or resampling. These have a negative effect on accuracy as balancing performance hurts the most common classes. 
Instead, some research has shown that an inverse square root class weight can help balance classes without overly punishing the dominant class [1]. 
To see this effect, we test set the weight WC for class C as the inverse x root of the class frequency FC .


$$
W_C = \frac{1}{\sqrt[x]F_C}
$$

As x approaches 1, we approach a class weight scheme of inverse class frequency also called “balanced” accuracy in scikit-learn. (Fig 5) 
We can see that values close to 1 decrease accuracy by overly punishing the dominant class, while larger root values don’t punish enough, 
and the best accuracy is at root 2. 

<img width="636" height="483" alt="image" src="https://github.com/user-attachments/assets/08429b57-ba2e-4cae-a970-4f50c1a6dbec" />\
Fig 5: Varying inverse x root class frequency weight. 
The best accuracy (0.524) is at root 2 (inverse square root class weight).


This pattern can be further investigated by analyzing the inverse x root on precision and recall (Fig 6). 
At x values close to 1 the model is weighted against defaulting to the dominant class, but recall suffers on the dominant class.

<img width="678" height="534" alt="image" src="https://github.com/user-attachments/assets/d01d0ad6-14ea-421a-a1ba-b479c31d5d9e" />\
Fig 6: Precision and recall for the highest (24), middle (11), 
and lowest (0) frequency classes as a function of the inverse x root.

------------------------------------------------------------------------

## Final Model

The final model configuration:

-   Logistic Regression
-   L2 regularization
-   Regularization strength = 18
-   Inverse square root class weighting
-   TF-IDF feature scaling

------------------------------------------------------------------------

## Reference
 Mahajan, D., Girshick, R., Ramanathan, V., He, K., Paluri, M., Li, Y., Bharambe, A., & Van Der Maaten, L. (2018). 
 Exploring the limits of weakly supervised pretraining. Lecture Notes in Computer Science, 185–201. https://doi.org/10.1007/978-3-030-01216-8_12
