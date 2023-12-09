import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('data/Spam Email raw text for NLP.csv')
df = df.drop('FILE_NAME', axis=1)
df['MESSAGE'] = df['MESSAGE'].replace('[^\w]+', ' ', regex=True)

nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('english')

df['MESSAGE'] = df['MESSAGE'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stopword]))

tf = TfidfVectorizer(ngram_range=(1, 3), max_features=2500)
X = tf.fit_transform(df["MESSAGE"]).toarray()
y = df["CATEGORY"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=1, stratify=y)

model = MultinomialNB()
model.fit(x_train, y_train)

train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

# print(classification_report(train_pred, y_train))
print(classification_report(test_pred, y_test))
