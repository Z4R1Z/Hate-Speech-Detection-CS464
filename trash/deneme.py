from tkinter import W
from trace import Trace
import pandas as pd
import numpy as np
import nltk
import pickle
import csv
#nltk.download()

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datasets import load_dataset

dataset = load_dataset("jigsaw_toxicity_pred")
print( dataset)

np.random.seed(500)
data = pd.read_csv("train_E6oV3.csv")

data['tweet'].dropna(inplace=True)
data['tweet'] = [entry.lower() for entry in data['tweet']] # all lowercase
data['tweet']= [word_tokenize(entry) for entry in data['tweet']] # tokenization
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ #seperate adjectives
tag_map['V'] = wn.VERB #seperate verbs
tag_map['R'] = wn.ADV #seperate adverbs

for index,entry in enumerate(data['tweet']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    data.loc[index,'text_final'] = str(Final_words)
    
print("Finished preprocessing")

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(data['text_final'],data['label'],test_size=0.3)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(data['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

""" further improvement
Tfidf_vect = pd.read_pickle("vectorizer_train.pk")

#with open('vectorizer_train.pk', 'wb') as file:
#    pickle.dump(Tfidf_vect, file)

#with open("Train_X.csv","w", newline='') as f:
    #writer = csv.writer(f)
    #writer.header()
    #writer.writerows(Train_X_Tfidf)
#Naive.save_weights("MNB.h5")
"""

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
predictions_NB = Naive.predict(Test_X_Tfidf)
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)


MNB_json = Naive.to_json()
with open("MNB.json","w") as json_file:
    json_file.write(MNB_json)
Naive.save_weights("MNB.h5")

#SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

SVM_json = SVM.to_json()
with open("SVM.json","w") as json_file:
    json_file.write(SVM_json)
SVM.save_weights("SVM.h5")


#RandomForest
RFC = RandomForestClassifier()
RFC.fit(Train_X_Tfidf, Train_Y)
predictions_RFC = RFC.predict(Test_X_Tfidf)
print("RFC Accuracy Score -> ", accuracy_score(predictions_RFC, Test_Y)*100)


RFC_json = RFC.to_json()
with open("RFC.json","w") as json_file:
    json_file.write(RFC_json)
RFC.save_weights("RFC.h5")
