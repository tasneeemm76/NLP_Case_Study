dataset = pd.read_csv("depression_dataset_reddit_cleaned.csv")

stop_words = set(stopwords.words('english'))
port_stemmer = PorterStemmer()

def preprocess (text):
    text = re.sub(r'\W', ' ', text)  # special characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # single characters
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)  # Remove single characters from the start
    text = re.sub(r'\s+', ' ', text, flags=re.I) 
    text = text.lower() 
    text = ' '.join([port_stemmer.stem(word) for word in text.split() if word not in stop_words]) 
    return text

dataset['clean_text'] = dataset['clean_text'].apply(preprocess_text)

tdif = TfidfVectorizer(max_features=3000)
X = tdif.fit_transform(dataset['clean_text']).toarray()
y = dataset['is_depression']

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

NaiveBayes_classifier = MultinomialNB()
NaiveBayes_classifier.fit(X_train, y_train)

#evaluating our model
y_pred = NaiveBayes_classifier.predict(X_test)
result = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)



LSTM ALGORITHM
# Import necessary libraries

data = pd.read_csv("depression_dataset_reddit_cleaned.csv")
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X = tfidf_vectorizer.fit_transform(data['clean_text']).toarray()
y = data['is_depression']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vocab_size = X.shape[1]  # Vocabulary size based on TF-IDF features
max_len = X.shape[1]  # Maximum length of input sequence

# Create and compile the LSTM model
lstm_model = Sequentials()
lstm_model.add(Embedding(input_dims=vocabulary_size, output_dims=128, input_length=max_len))
lstm_model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
lstm_model.add(Dense(units=1, activation='sigmoid'))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Convert X_train and X_test to sequences and pad them to ensure uniform length
X_train_seq = pad_sequences(X_train, maxlen=max_len)
X_test_seq = pad_sequences(X_test, maxlen=max_len)

lstm_model.fit(X_train_seq, y_train, epochs=5, batch_size=32)

# Evaluate the LSTM model
lstm_scores = lstm_model.evaluate(X_test_seq, y_test)
accuracy_lstm = lstm_scores[1]

# Print accuracy for LSTM
print("\nLSTM Accuracy:", accuracy_lstm)

