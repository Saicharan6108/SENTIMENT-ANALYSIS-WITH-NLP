# SENTIMENT-ANALYSIS-WITH-NLP

*COMPANY*: CODTECH IT SOLUTIONS  
*NAME*: MOLIGE SAI CHARAN  
*INTERN ID*: CT04WY43  
*DOMAIN*: MACHINE LEARNING  
*DURATION*: 4 WEEKS  
*MENTOR*: NEELA SANTOSH

Sentiment Analysis Using TF-IDF and Logistic Regression
Sentiment analysis, also known as opinion mining, is a natural language processing (NLP) technique used to determine whether data is positive, negative, or neutral. In this project, sentiment analysis is performed on a dataset of customer reviews to classify the sentiments as either positive or negative. The implementation uses Python, along with several machine learning and NLP libraries, to build, train, and evaluate a classification model.

Objective
The goal of the project is to classify customer reviews into positive or negative sentiments using machine learning. To achieve this, we use TF-IDF vectorization for feature extraction and Logistic Regression for classification.

Tools and Libraries Used
The following tools and Python libraries were used:

Python: The core programming language used for this project.

Pandas: Used to create and manipulate the dataset in the form of a DataFrame.

Scikit-learn (sklearn): A powerful machine learning library that provides tools for preprocessing, feature extraction, model building, training, and evaluation.

TfidfVectorizer: Converts raw text into numeric vectors using Term Frequency-Inverse Document Frequency.

train_test_split: Splits the dataset into training and test sets.

LogisticRegression: Used to build a classification model.

classification_report and accuracy_score: Used for evaluating the model's performance.

Jupyter Notebook: An interactive development environment ideal for machine learning workflows and documentation.

Dataset Preparation
The dataset consists of 200 customer reviews, evenly split between 100 positive and 100 negative reviews. Positive and negative samples were generated synthetically by combining a list of generic customer review sentences. Each review is labeled:

1 for positive sentiment

0 for negative sentiment

After creating the dataset, the positive and negative reviews were combined and shuffled to ensure that the model does not learn from any ordering patterns in the data.

Preprocessing and Feature Extraction
The raw text data cannot be directly fed into a machine learning model. Therefore, it needs to be transformed into a numerical format. This is done using TF-IDF Vectorization:

TF-IDF (Term Frequency-Inverse Document Frequency) transforms the text into numerical features based on how important a word is in a document relative to the entire corpus.

Common stop words (like “the”, “is”, “and”) are removed during vectorization to focus on meaningful words.

The resulting matrix from TF-IDF represents each review as a sparse vector, where each element corresponds to the weighted importance of a specific word in that review.

Model Training
After vectorization, the data is split into a training set and a test set using train_test_split. To ensure that both the positive and negative classes are fairly represented in both sets, the split is done with the stratify parameter.

A Logistic Regression model is then trained on the training data. Logistic regression is a classification algorithm that estimates the probability of a sample belonging to a specific class.

Evaluation
The trained model is evaluated on the test set using:

Accuracy Score: The proportion of correctly predicted samples.

Classification Report: Includes precision, recall, and F1-score for both classes (positive and negative).

These metrics give a clear understanding of how well the model performs. With a dataset of 200 reviews, the model has enough data to generalize and make meaningful predictions.

Conclusion
This project demonstrates how to apply NLP techniques and machine learning algorithms to perform sentiment analysis on text data. Using TF-IDF for feature extraction and Logistic Regression for classification is a straightforward yet effective approach. The tools from scikit-learn make it easy to build the pipeline from preprocessing to evaluation. With a well-prepared dataset and proper model evaluation, this method can be used in real-world applications like product reviews, social media monitoring, and customer feedback analysis.

#Output

