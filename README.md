# Random-Forest[Multi-classification ] Classifier
**Random Forest is a supervised machine learning algorithm which is based on ensemble learning. In this kernel, I build Random Forest multiclass-Classifier model to predict what kind of salary should person have. I have demonstrated the feature selection process using the Random Forest model to find only the important features, rebuild the model using these features and see its effect on accuracy. I have used the HR-Dataset for this project.**

__Table of Contents__

1.__Introduction to Random Forest algorithm__

Random forest is a supervised learning algorithm. It has two variations – one is used for classification problems and other is used for regression problems. It is one of the most flexible and easy to use algorithm. It creates decision trees on the given data samples, gets prediction from each tree and selects the best solution by means of voting. It is also a pretty good indicator of feature importance.

Random forest algorithm combines multiple decision-trees, resulting in a forest of trees, hence the name Random Forest. In the random forest classifier, the higher the number of trees in the forest results in higher accuracy.

2.__Random Forest algorithm intuition__

![image](https://user-images.githubusercontent.com/89013703/179548393-5db87644-c5de-45ca-9326-c07290b6ffe3.png)

3.__Advantages and disadvantages of Random Forest algorithm__

Advantages:-

.Random forest algorithm can be used to solve both classification and regression problems.

.It is considered as very accurate and robust model because it uses large number of decision-trees to make predictions.

.Random forests takes the average of all the predictions made by the decision-trees, which cancels out the biases. So, it does not suffer from the overfitting problem.

.Random forest classifier can handle the missing values. There are two ways to handle the missing values. First is to use median values to replace continuous variables and second is to find highly corelated variable and fill missing values with that value.

.Random forest classifier can be used for feature selection. It means selecting the most important features out of the available features from the training dataset.

Disadvantages:-

.The biggest disadvantage of random forests is its computational complexity.

.Random forests is very slow in making predictions because large number of decision-trees are used to make predictions.All the trees in the forest have to make a prediction for the same input and then perform voting on it. So, it is a time-consuming process.

.The model is difficult to interpret as compared to a decision-tree, where we can easily make a prediction as compared to a decision-tree.

4.__The problem statement__

In this kernel, I try to make predictions where the prediction task is to determine whether what kind of salary should person have. I implement Random Forest Classification with Python and Scikit-Learn. So, to answer the question, I build a Random Forest classifier to
predict what kind of salary should person have.

I have used the HR-Salary classification data set for this project.

5.__Feature selection with Random Forests__

Random forests algorithm can be used for feature selection process. This algorithm can be used to rank the importance of variables in a regression or classification problem.

We measure the variable importance in a dataset by fitting the random forest algorithm to the data. During the fitting process

Features which produce large values for this score are ranked as more important than features which produce small values. Based on this score, we will choose the most important features and drop the least important ones for model building.

6.__Difference between Random Forests and Decision Trees__

I will compare random forests with decision-trees. Some salient features of comparison are as follows:-

.Random forests is a set of multiple decision-trees.

.Decision-trees are computationally faster as compared to random forests.

.Deep decision-trees may suffer from overfitting. Random forest prevents overfitting by creating trees on random forests.

.Random forest is difficult to interpret. But, a decision-tree is easily interpretable and can be converted to rules.


7.__Important Hyperparameters__

Hyperparameters are used in random forests to either enhance the performance and predictive power of models or to make the model faster.

Following hyperparameters increases the predictive power:

1. n_estimators– number of trees the algorithm builds before averaging the predictions.

2. max_features– maximum number of features random forest considers splitting a node.

3. mini_sample_leaf– determines the minimum number of leaves required to split an internal node.

4. max_depth = max number of levels in each decision tree

5. min_samples_split = min number of data points placed in a node before the node is split













