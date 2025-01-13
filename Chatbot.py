import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

import json
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

# Initialize Lemmatizer and load data
lemmatizer = WordNetLemmatizer()
with open('data.json') as file:
    data = json.load(file)


# Preprocess data
patterns = []
tags = []
responses = {}
for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        patterns.append(tokens)
        tags.append(intent['tag'])
    responses[intent['tag']] = intent['responses']

# Lemmatize and encode
words = [lemmatizer.lemmatize(w.lower()) for p in patterns for w in p if w.isalnum()]
words = sorted(set(words))
tags = sorted(set(tags))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Prepare training data
training_sentences = [" ".join(p) for p in patterns]
training_labels = tags
label_encoder = LabelEncoder()
training_labels = label_encoder.fit_transform(tags)

# Convert to bag-of-words
def bag_of_words(sentence, words):
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence)]
    return np.array([1 if w in sentence_words else 0 for w in words])

X_train = np.array([bag_of_words(" ".join(p), words) for p in patterns])
y_train = np.array(training_labels)

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(tags), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def get_sentiment(sentence):
    score = sia.polarity_scores(sentence)
    if score['compound'] > 0.2:
        return "Positive"
    elif score['compound'] < -0.2:
        return "Negative"
    else:
        return "Neutral"

from flask import Flask, request, jsonify
import random
from tensorflow.keras.models import load_model

# Load the trained model and other data
model = load_model('chatbot_model.h5')

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    bow = bag_of_words(user_message, words)
    prediction = model.predict(np.array([bow]))[0]
    tag = tags[np.argmax(prediction)]

    # Get a random response for the tag
    response = random.choice(responses[tag])

    # Analyze sentiment
    sentiment = get_sentiment(user_message)

    return jsonify({"response": response, "sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)

