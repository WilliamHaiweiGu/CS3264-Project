import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
import re

# Load Data
df = pd.read_csv('train.csv')  # Ensure train.csv is in the same directory as the script

# Select Relevant Columns
df = df[['tweet', 'class']].dropna()  # Keep only 'tweet' and 'class', and remove missing values

# # Text Cleaning Function
# def clean_text(text):
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'http\S+', '', text)  # Remove URLs
#     text = re.sub(r'@\w+', '', text)  # Remove @mentions
#     text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
#     return text.strip()
def clean_text(text: str) -> str:
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ', text)
    return re.sub(r" +", ' ', text)

df['clean_tweet'] = df['tweet'].apply(clean_text)  # Apply text preprocessing
print("Sample cleaned tweet:", df['clean_tweet'].iloc[0])  # Print a sample cleaned tweet for verification
# Check for Class Distribution
print("Class distribution:\n", df['class'].value_counts())  # Display the distribution of classes

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(df['clean_tweet'], df['class'], test_size=0.2, random_state=42)

# Enhanced pipeline with comprehensive improvements
enhanced_pipeline = Pipeline([
    ('vect', CountVectorizer(
        stop_words='english',
        ngram_range=(1, 3),  # Try 1-3 grams
        max_features=10000,  # Limit number of features
        min_df=2, max_df=0.95  # Filter rare and overly common words
    )),
    ('tfidf', TfidfTransformer(
        norm='l2',  # Use L2 normalization
        use_idf=True,
        smooth_idf=True
    )),
    ('clf', MultinomialNB(
        alpha=0.1,  # Try smaller alpha
        fit_prior=True  # Use class prior probabilities from data
    )),
])

# Cross-validation to find optimal parameters
from sklearn.model_selection import GridSearchCV

parameters = {
    'vect__ngram_range': [(1,1), (1,2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (0.1, 0.5, 1.0),
}

grid_search = GridSearchCV(enhanced_pipeline, parameters, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
y_pred = grid_search.predict(X_test)

# Evaluate Model Performance
print(classification_report(y_test, y_pred))
