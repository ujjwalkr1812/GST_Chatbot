# Natural Language Processing
# Importing the libraries
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('questionsVSclass.csv')
resultset = pd.read_csv('classVSanswers.csv')
# Cleaning the texts
import re
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 300):
    review = re.sub('[^a-zA-Z]', ' ', dataset['question'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
# Predicting the Test set results
while(True):
    x_test = input("Enter your question: ")
    if(x_test=="quit"):
        break
    only_char = re.sub('[^a-zA-Z]',' ',x_test)
    lower_x_test = only_char.lower()
    list_x_test = lower_x_test.split()
    ps2 = PorterStemmer()
    stemmed_x_test = [ps2.stem(word) for word in list_x_test if not word in set(stopwords.words('english'))]
    preprocess_x_test = ' '.join(stemmed_x_test)
    final_x_test = cv.transform([preprocess_x_test]).toarray()
    y_pred_x = classifier.predict(final_x_test)
    print("Answer: "+resultset['answer'][y_pred_x])
