import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('data/Spam Email raw text for NLP.csv')
print(df['FILE_NAME'])
