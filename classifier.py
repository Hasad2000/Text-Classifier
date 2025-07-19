import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

df = pd.read_csv('labelled_data_set.csv')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    tokens = word_tokenize(text.lower())
    cleaned = [w for w in tokens if w.isalpha() and w not in stop_words]
    return ' '.join(cleaned)

df['cleaned_text'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
Y = df['label']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25)

model = LogisticRegression()
model.fit(X_train, Y_train)

#predictions = model.predict(X_test)
#print("Classification Report:")
#print(classification_report(Y_test, predictions))


print("\n📢 Ready to classify text as 'mean' or 'kind'. Type 'quit' to exit.")
while True:
    user_input = input("\nEnter a sentence: ")
    if user_input.lower() == 'quit':
        break
    cleaned_input = clean_text(user_input)
    vectorized_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(vectorized_input)
    print(f"👉 Prediction: {prediction[0]}")