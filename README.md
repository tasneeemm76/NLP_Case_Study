NATURAL LANGUAGE PROCESSING 
Problem statement:-
"Predictive Modelling for Detecting Depression Symptoms in Social Media Text"

The solution for our problem statement includes creating a predictive model for social media post. We’ve used a cleaned dataset (represented by the 'clean_text' column) and classify if the text indicates symptoms of depression ('is_depression' column).

Objective:-
The objective is to build a machine learning model capable of detecting signs of depression in social media text data. By accurately identifying the signs, early intervention and support can be provided to individuals who may be at risk or in need of assistance.

Methodology: Outline the NLP techniques and machine learning algorithms to be employed for feature extraction, model training, and evaluation. This could include preprocessing steps (e.g., tokenization, stop word removal, stemming/lemmatization), feature representation (e.g., TF-IDF vectors), and classification algorithms (e.g., Naive Bayes, SVM, neural networks).

OUPUT
The output includes accuracy score, confusion matrix, and classification report
It utilizes naïve bayes, KNN, Logistic regression and LSTM algorithms to study the model

Accuracy:  it determines the correctness of the model of predicting the symptoms of depressions properly.
In this case, the accuracy is approximately 90.63%. this  means that the it correctly predicts whether a social media post indicates depression symptoms around 90.63% of the time.

Confusion Matrix: It is a table that indicates the classification algorithm’s performance.
It has four terms:
1.	True Positives (TP)
2.	True Negatives (TN)
3.	False Positives (FP)
4.	False Negatives (FN).
In this confusion matrix:
True Positives (TP): 739, meaning 739 instances where depression symptoms were present.
True Negatives (TN): 663 examples where depression symptoms were not present.
False Positives (FP): 120 cases as depression symptoms when they were not.
False Negatives (FN): 25 cases as not having depression symptoms when they did.

Classification Report: The classification report is a overview of different evaluation metrics like precision, recall, and F1-score for each class (0 and 1 in this case).
In this classification report:
For class 0 (indicating no depression symptoms):
Precision: 0.96, Recall: 0.85, F1-score: 0.90
in class 1 (indicating depression symptoms):
Precision: 0.86,Recall: 0.97,F1-score: 0.91
By analysing the accuracy, confusion matrix, and classification report,the models correctness is evaluated in detecting depression symptoms by social media text. The high accuracy score and balanced precision, recall, and F1-scores indicate that the model is effective in identifying both positive and negative instances of depression symptoms. This information helps in assessing the model's performance and its application in solving the problem of detecting depression symptoms in social media text.


Process of the LSTM model
1.	Epochs 1-5:
•	For every epoch, the model goes through the entire training dataset (X_train_seq and y_train) in batches of size 32.
•	After each epoch, the model's performance is checked on the validation 
•	Epoch 1:
•	The model starts with a loss of approximately 0.6941 and an accuracy of around 0.5003.
•	Epoch 2-5:
•	The loss and accuracy values fluctuate slightly across epochs, indicating that the model is learning but may not be converging optimally.
2.	Evaluation:
•	After the training process completes, the model's performance is verified on the test dataset (X_test_seq and y_test).
•	The evaluation results show a loss of approximately 0.6931 and an accuracy of around 0.5061.
•	The evaluation process takes about 33 seconds to complete.
Overall, the LSTM model achieves an accuracy of approximately 50.61% on the testing dataset. However, the model's efficiency may not be satisfactory, as the accuracy is close to random guessing (50%). The loss values around 0.6933 indicate that the model may not be learning effectively, and further optimization may be needed to improve its performance.

Output LSTM:
1.	Logistic Regression:
•	The correctness of the Logistic Regression model is approximately 95.02%, which means that it correctly predicts depression symptoms for 95.02% of the times in the test dataset.
    Classification Report:
•	Precision: for class 0 (non-depression) is 93%, which means that among the cases predicted as non-depression, 93% are actually non-depression.
•	Precision for class 1 (depression) is 98%, indicating that among the instances predicted as depression, 98% are actually depression.
•	Recall: for class 0 is 98%, meaning that 98% of the actual non-depression instances are correctly predicted as non-depression.
•	Recall for class 1 is 92%, indicating that 92% of the actual depression instances are correctly predicted as depression.
•	The F1-score, is the harmonic mean of precision and recall, is approximately 95% for both classes.
•	Support: The number of instances in each class is 783 for class 0 and 764 for class 1.
•	Summary: Overall, the Logistic Regression model performs very well, with high precision, recall, and F1-score for both classes, resulting in a high exactness.

2.	LSTM:
•The accurateness of the LSTM model is much lower at approximately 50.61%, indicating that it performs poorly in predicting depression symptoms compared to the Logistic Regression model.
•The LSTM model's accuracy is close to random guessing (50%), indicating that it is not very efficient in predicting the symptoms
•The LSTM model's poor performance could be because of lesser number of epochs which can be corrected in future.

  

