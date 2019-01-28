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

unique_words = cv.get_feature_names()
n_bwm = len(unique_words)

# Naive Bayesian Classifier Probabilities

keys = []
for i in range(0,60):
    for word in unique_words:
        keys.append(word+"0"+str(i))
        keys.append(word+"1"+str(i))

#keys[5][:-2]


prob = []
for i in range(0,60):
    for j in range(0,n_bwm):
        count0 = 0
        count1 = 0
        #For each type of question we have 5 different questions
        for k in range(5*i,5*i+5):
            if(X[k][j]==0):
                count0 += 1
            else:
                count1 += 1
        prob0 = (count0+1)/(5+n_bwm)
        prob1 = (count1+1)/(5+n_bwm)
        prob.append(prob0*100)
        prob.append(prob1*100)

mapper_out = dict()
for i in range(len(keys)):
    mapper_out[keys[i]] = prob[i]




# # Splitting the dataset into the Training set and Test set
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
# # Fitting Naive Bayes to the Training set
# from sklearn.naive_bayes import MultinomialNB
# classifier = MultinomialNB()
# classifier.fit(X_train, y_train)
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# y_pred = classifier.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)


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
    cv2 = CountVectorizer()
    final_x_test_bwm = cv2.fit_transform([preprocess_x_test]).toarray()
    unique_words2 = cv2.get_feature_names()
    n_bwm_2 = len(unique_words2)
    prob_each_class = []
    for i in range(0,60):
        prior_prob = (5+1)/(300+60)
        prob = prior_prob
        for j in range(0,n_bwm):
            if(unique_words[j] in unique_words2):
                prob = prob*mapper_out.get(unique_words[j]+"1"+str(i))
            else:
                prob = prob*mapper_out.get(unique_words[j]+"0"+str(i))
        prob_each_class.append(prob)
    max_prob = max(prob_each_class)
    #index of max_probable class
    for i,x in enumerate(prob_each_class):
        if(x == max_prob):
            y_pred_x = i
    print("Answer: "+resultset['answer'][y_pred_x])
