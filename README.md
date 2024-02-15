# Churn Segmentation Modelling ANN
![Python 3.7](https://img.shields.io/badge/Python-3.7-yellow)  ![Tensforflow 2.7.0](https://img.shields.io/badge/Tensorflow-2.7.0-blue) ![Val_Acc](https://img.shields.io/badge/Validation%20Acc.-87%25-orange)


This is a complete Project that revolves around churn modelling and it contains every aspect from data cleaning down to model deployment. The data of a bank was used in this implementation and for modelling purposes an Artificial Neural Network was trained and used to predict the probability that a given customer would leave the bank(With 87% accuracy) and for deployment an API was developed which can be used for single prediction as well as batch prediction for a number of customers

> 'Churn’ refers to the rate at which a subscription company loses its subscribers because of subscription cancellations or elapses. This leads to loss of revenue. Churn rates really matter for subscription businesses because they are an important indicator of long term success.  
# Highlights of the Project

- #### Data Cleaning
    - #### Outlier Detection
    - #### Skewness
- #### Exploratory Data Analysis
- #### Feature Engineering
- #### Model Development
    - #### Hyperparameter tuning using GridSearchCV
    - #### Bias and Variance Analysis using Cross Validation Score
    - #### Validation and Evaluation
- #### Model Deployment using Flask 

# Walkthrough:

## 1. About the Data:

### All the Data used in this Project is from Kaggle Churn Modelling Dataset  

- [Kaggle Dataset: Churn Modelling](https://www.kaggle.com/shrutimechlearn/churn-modelling)  

### File Description

- Churn_Modelling_Original.csv(The fulll dataset without any preprocessing)

### Data Fields

You can find more about the data [Here](https://www.kaggle.com/shrutimechlearn/churn-modelling)

## 2. Data Cleaning: ([Data Cleaning Notebook](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Data_Cleaning_and_EDA.ipynb))

#### Outlier Detection:

> Box plots were used for the initial assesment of the data which concluded that two features 'Age' and 'CreditScore' might have outliers

![Box Plots](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Visualizations/Box_Plots.png)  

> Hence, I only considered these two features while treating for outliers.  
Also, we have a various methods to find the outliers. In this project i've used the IQR method but other options are:

    1. Z-score method
    2. Robust Z-score
    4. Winterization method(Percentile Capping)
    5. DBSCAN Clustering
    6. Isolation Forest
    
> Two data points were identified as outliers and were hence removed

![outliers](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/_images/Outliers.png)

#### Skewness:

> Sample Skewness of each feature was calculated using the scipy stats.skew function and was plotted for analysis along with that QQ plots were studied

Skewnedd Plot:

![Skewness](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Visualizations/Skewness_Plot.png)

QQ Plot: 

![QQ Plots](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Visualizations/QQ_Plots.png)

> The Study showed both of the two continuous variable 'Balance' and 'EstimatedSalary' were skewed, And since we have to scale our data to fit an ANN we can use a StandardScalar that will not only scale the data but also standardize it. And hence, we don't have to treat the data for skewness as Standardization will negative a lot of this skewness.

## 3. Exploratory Data Analysis(EDA): ([EDA Notebook](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Data_Cleaning_and_EDA.ipynb))

> For getting insights from the data various plots like Histograms, Pivoted Histograms and heatmaps were created

#### Histogram for Categorical Variables:
> Categorical variables: 'Geography', 'Gender', 'NumberofProducts', 'HasCrCrad', 'IsActiveMember', 'Exited'(Target variable/Label)

![Histogram for Categorical Variables](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Visualizations/Histograms.png)

#### Histogram for Pivot Data of Categorical variables on the Target Variable(Exited):

![Histogram for Pivot Data of Categorical variables on the Target Variable](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Visualizations/Histogram_againts_Exited.png)

#### Correlation Matrix:

![Correlation Matrix](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Visualizations/Correlation_Matrix.png)

#### Observations from EDA:

> 1. The Distribution of the Target variable is highly imbalanced, out of 10000 instances 7961 were negative samples or sample were the customer did not leave the bank which translate to a Negative to Postive imbalance of alomst 0.79 - 0.21, which means that even a naive classifier(only gives negative prediction) will reach an accuracy of 79% and hence the baseline accuracy for evaluation can be set at 80%.  

> 2. The histograms of variables when pivoted around the target variable shows that there are variable which have highly imbalance pivot distribution for example 'NumofProducts' and 'Gender'. Such distribution are both good and bad for Machine Learning algorithms, good because such features can be used to create a clearer distinction between the classes by the classifier and bad because models tends to overfit to the training data due to such distribution and hence a dropout layer which will reduce overfitting is necessary while training the ANN.  

> 3. The Correlation Matrix higlights the point that there are no feature in the dataset with a high correlation(Greater than 0.5) with the target variable 'Exited'. And hence, Traditional ML Models are expected to not give good results. This fact was the whole reason behind the use of ANN for this particular problem and dataset in this project.

## 4. Feature Engineering: ([Feature Engineering Notebook](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Feature_Engineering_and_Model_Building.ipynb))

### Categorical Variable Encoding

#### Mapping:  
>Since 'Gender' was a binary variable it was encoded with the help of the Scikit-Learn LabelEncoder:

    Male ----> 1
    Female ----> 2
    
> The variable 'Geography' had more than 2 categories it was encoded using Scikit-Learn OneHotEncoding and subsequently one of the column was droped to recover from the dummy trap:

    France ----> [0,0]
    Spain ----> [0,1]
    Germany ----> [1,0]
    
### Standardization

>Standardizing a dataset involves rescaling the distribution of values so that the mean of observed values is 0 and the standard deviation is 1.  
This can be thought of as subtracting the mean value or centering the data.

Standardization assumes that your observations fit a Gaussian distribution (bell curve) with a well-behaved mean and standard deviation

A value is standardized as follows:

![Z-Score](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/_images/Standardization.png)

Where the mean is calculated as:

![Mean](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/_images/Mean.png)

And the standard_deviation is calculated as:

![Standard Deviation](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/_images/Standard_Deviation.png)

## 5. Modell Building: ([Modell Building Notebook](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Feature_Engineering_and_Model_Building.ipynb))

> While building the ANN, i will the sklearn function 'GridSearchCV' for tuning the hyperparameter of the ANN but since the ANN is made through keras library with tensorflow backend the model would not be compatible with sklearn.  
To resolve this problem we have to use the sklearn wrapper given in the kears library that will take the keras ANN object and gives out a Classifier object that is compatible with the sklearn library and then we can use the GridSearchCV function

To Use the keras wrapper for sklearn 'KerasClassifier'(You can read more about the wrapper [Here](https://faroit.com/keras-docs/1.0.6/scikit-learn-api/))   
We have to create the architecture of the ANN inside a builder function which will be passed inside the wrapper to generate the sklearn compatible ANN classifier.

You can look at the build function and the wrapper implementation as well as GridSearchCV implementation in my project in this [notebook](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Feature_Engineering_and_Model_Building.ipynb)


### ANN Architecture:

After experimenting with various configuration of number of neurons and number of hidden layers, I settled with a Architecture that containe:

    Number of Hidden Layers = 3
    Number of Dropout Layers = 3(After Each Hidden Layer)
    Dropout paramter = 0.1
    Number of Neurons:
    In First Layer = 15 Neurons
    In Second Layer = 25 Neurons
    In Third Layer = 15 Neurons
    
 ![ANN Architecture](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/_images/ANN_Architecture.png)
    

### Grid Search:

Parameter Space:

> Optimizer = ['adam', 'rmsprop'], Epochs = [100, 200], Batch_Size = [32, 64, 128]

Result:

> Best Parameters: Optimer = 'adam', Batch Size = 32, Epochs = 200  
> Accuracy on Train Set with best parameters = 86%

## 6. Validation & Evaluation: ([Validation Notebook](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Validation.ipynb))

### Cross Validation:

> To Evaluate the bias and variance in our model we can use the cross val score method from sklearn to test our model on K-Folds, Doing so will give us a measure of variance in the model accuracy and as well as a measure of central tendency of the accuracy which will be a better approximation of the performance of the model on the training data.

20 K-Fold Results:

    The Mean Accuracy: 0.8498
    The Standard Deviation of the sample: 0.0187
    The Variance of Accuracy: 0.0003492
    
### Bias & Variance Analysis:

The Mean Accuracy is about 85% which means we have a medium to low bias.
The Standard Deviation is 0.0187 and accordingly we have a variance of 0.0003492 or about 0.0004 which reflects that our model shows a low variance as well.

![Bias Variance](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/_images/Bias-Variance.png)

The models shows good results on the training data and is a potential model for segmentation but we still have to check for performance on the test data.

### Evaluation on Test Data:

>The Accuracy reached on the Training data is: 87% and The Accuracy on the test data is: 86%
The shows that there is no overfitting in the model on the train data since the accuracies on both the train set and the test set is comparable.
This is because in our ANN architecture we have implemented a 'Dropout' Layer after every hidden layer that reduces overfitting.

### Confusion Matrix:

![Confusion Matrix](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Visualizations/Confusion_Matrix.png)


Everyting thing is good except that our model has quite a lot of False Negatives(232) which means that there are 232 such customers in the test set that were predicted to stay by our model but in actuality they exited.  

Such high number of False Positive decrease the models 'Recall' or 'Senstivity', we can calculate some other evaluation criteria so that we can better judge our model. 

>We can calculate the ratio's(Criterias) by using the sklearn librabry but here i am manually calculating them as this is a nice practice by which i can remember both the mathematical relation and what the ratio signifies

Evaluation Criteria/Ratio's
    
    The Precision(Positive Predictive value) of the Model: 0.8044
    (The precentage of correct Positive Predictions out of all Positive Predictions)
    
    The Recall(True Positive rate) of the Model: 0.4383
    (The percentage of Positive cases that were correctly identified out of the total Positive cases)
    
    The F1 Score of the Model: 0.5674
    (Harmonic mean of precision and recall)
    

### Precision Recall Curve:

> A precision-recall curve shows the relationship between precision (= positive predictive value) and recall (= sensitivity) for every possible cut-off. A precision-recall curve helps to visualize how the choice of threshold affects classifier performance, and can even help us select the best threshold for a specific problem.

![Precision Recall Curve](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Visualizations/Precision-Recall_Curve.png)


> Looking at the Precision Recall Curve there isn't any threshold value at which both Precision and Recall will be greater than 70%. Hence, we have trade off one of the two. Here for this problem the bank must focus on reducing the Flase Positives so that the subset of customers at risk of churning can be reduced so that the bank official have less people to analyse or tackle.   

So, keeping that in mind we can see that reducing False Positive is of greater importance than reducing Flase Negative and this trade translate to having a higher precision than recall. Hence, our original threshold of 0.6 can also be used as a good threshold even at low recall due to high precision.

### Inferences drawn from Evaluation of the model:

As, we can see even if the accuracy is high that does not mean that the model is perfect, the overall precision was nice at 80% but the recall or sensitivity that is the power of the model to find the Positive cases from the data is low at 43.8%.

So, a note for future improvement can be to tweak the model so that the recall can be improved and this can be done by a number of ways but the most straightforward of them all is by simply replacing the scoring parameter of the grid search to F1 Score that will take into account both Precision as well as Recall.

## 7. Model Deployment:

Using Flask library an API is developed that can be used to get prediction for a single customer or a batch of customers, the data can be fed into the API in a JSON file with predifined structure(Schema). For More information about the API and the JSON Schema you see the [Model Deployment](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/tree/master/Model_Deployment) Directory of this project.

The API contains three route/method:

    1. /Help : Takes no arguments and return a JSON file whose '.text' method contains informatino about the API and it's usage.
    
    2. /predict : Takes a JSON file as argument that must conatins all feature values of a customer for which 
    prediction is required in a predifined format and return a JSON file with the required prediction and probability of churn
    
    3. /batch_predict : Takes a JSON file as argument that must containes all feature values of multiple customers for 
    which prediction are required and returns a JSON file with the reuired Predictions and respectivy probabilities of churn
    
    
## Conclusion of the Project:

An ANN approach was used to modell the churn rates of a bank, after cleaning and transforming the raw data a Grid Search was immplemented to find the best parameters for the model which were:

    ANN Architecture:

      Number of Hidden Layers = 3
      Number of Dropout Layers = 3(After Each Hidden Layer)
      Dropout paramter = 0.1
      Number of Neurons:
        In First Layer = 15 Neurons
        In Second Layer = 25 Neurons
        In Third Layer = 15 Neurons
     
    Best paramters:
     
      Optimizer = adam
      Epochs = 200
      Batch Size = 32
     
The Model was then evaluated with K-Fold validation method, The result from 20 K-Folds were:
 
    The Mean Accuracy: 0.8498
    The Standard Deviation of the sample: 0.0187
    The Variance of Accuracy: 0.0003492
    
Evaluation on the test data yields:

    Accuracy on Test Data = 87%
    Precision = 80.4%
    
The Model Having high Accuracy as well as Precision is considered to be a good solution for the problem of modelling churn, The model can be used for Segmentation as well as Classification.
