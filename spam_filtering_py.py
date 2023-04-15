import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
import re
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv("spam.csv",encoding="latin")
df.head()

df.info()
df.isna().sum()

df.rename({"v1":"label", "v2":"text"},inplace=True,axis=1)
df.tail()

le=LabelEncoder()
df['label']=le.fit_transform(df['label'])

nltk.download("stopwords")
corpus = []
length=len(df)
ps = PorterStemmer()

for i in range(0,length):
    text = df['text'][i]
    text = re.sub('[^a-zA-Z]', ' ', text)
    text=text.lower()
    text = text.split()
    pe=PorterStemmer()
    stopword = stopwords.words("english")
    text = [ps.stem(word) for word in text if not word in set(stopword)]
    text = ' '.join(text)
    corpus.append(text)
corpus

cv = CountVectorizer(max_features = 35000)
X = cv.fit_transform(corpus).toarray()
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.20, random_state = 0)
sm = SMOTE(random_state = 2)

X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

print( 'After Oversampling, the shape of train X: {}'.format(X_train_res.shape))
print( 'After Oversampling, the shape of train y: {} \n'.format(y_train_res.shape))

print("After Oversampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After oversampling, counts of label '0': {}".format(sum(y_train_res == 0)))
pickle.dump(cv, open('cv1.pkl', 'wb'))

df.describe()

df.shape

df["label"].value_counts().plot(kind="bar", figsize=(12,6))
plt.xticks(np.arange(2),('Non spam','spam'), rotation=0);

sc = StandardScaler()
x_bal = sc.fit_transform(X)

x_bal =pd.DataFrame(x_bal)

model = MultinomialNB()
model.fit(X_train_res,y_train_res)

model = Sequential()

X_train.shape

model.add(Dense(units = X_train_res.shape[1],activation="relu",kernel_initializer="random_uniform"))
model.add(Dense(units = 100,activation="relu",kernel_initializer="random_uniform"))
model.add(Dense(units = 100,activation="relu",kernel_initializer="random_uniform"))
model.add(Dense(units = 1,activation="sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
generator = model.fit(X_train_res,y_train_res,epochs=10,steps_per_epoch=len(X_train_res)//64)

y_pred = model.predict(X_test)
y_pred

y_pr = np.where(y_pred>0.5,1,0)
y_test

cm = confusion_matrix(y_test, y_pr)
score = accuracy_score(y_test, y_pr)
print(cm)
print('Accuracy Score Is:- ',score*100)

def new_review(new_review):  
  new_review = new_review
  new_review = re.sub('[^a-zA-Z]',' ',new_review)
  new_review = new_review.lower()
  new_review = new_review.split()
  ps=PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  new_review = [ps.stem(word) for word in new_review if not word in  set(all_stopwords)]
  new_review = ' '.join(new_review)
  new_corpus = [new_review]
  new_X_test = cv.transform(new_corpus).toarray()
  print(new_X_test)
  new_y_pred = model.predict(new_X_test)
  print(new_y_pred)
  new_X_pred = np.where(new_y_pred>0.5,1,0)
  return new_y_pred
new_review = new_review(str(input("Enter New Review...")))
threshold = 0.5
y_pred = (y_pred > threshold).astype(int)

cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
print(cm)
print('Accuracy Score Is:- ',score*100)

model.save('spam.h5')