import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
import nltk
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from joblib import Memory

# Download necessary NLTK data files (if not already downloaded)
nltk.download('punkt_tab')
nltk.download('vader_lexicon')

memory = Memory(location='cache_dir', verbose=0)

def clean_text(text: str) -> str:
    text = re.sub(r"@[A-Za-z0-9_]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^A-Za-z.!?'0-9]", ' ', text)
    text = re.sub(r'\t', ' ', text)
    return re.sub(r" +", ' ', text)

def extract_mwes_from_corpus(corpus, top_n=50, freq_filter=3):
    tokens = []
    for text in corpus:
        tokens.extend(nltk.word_tokenize(text.lower()))
    finder = BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(freq_filter)
    bigram_measures = BigramAssocMeasures()
    top_bigrams = finder.nbest(bigram_measures.pmi, top_n)
    return top_bigrams

class IncendiaryContentFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 swear_words=None, 
                 hate_speech_words=None, 
                 extremist_words=None,
                 mwe_list=None):
        self.swear_words = frozenset(swear_words) if swear_words else frozenset({
            'fuck', 'shit', 'damn', 'bitch', 'asshole', 'crap', 
            'dick', 'piss', 'cunt', 'cock', 'prick', 'bastard', 'slut', 'whore'
        })
        self.hate_speech_words = frozenset(hate_speech_words) if hate_speech_words else frozenset({
            'nigger', 'nigga', 'chink', 'spic', 'kike', 'faggot', 'fag', 'dyke', 'gook', 
            'wetback', 'towelhead', 'camel jockey', 'cracker', 'libtard'
        })
        self.extremist_words = frozenset(extremist_words) if extremist_words else frozenset({
            'jihad', 'isis', 'al-qaeda', 'extremist', 'martyr', 'revolution', 'uprising', 
            'caliphate', 'sharia', 'takfir', 'khilafah', 'jihadi', 'terrorist', 'suicide bomber'
        })
        self.mwe_list = mwe_list if mwe_list else []
        self.sia = SentimentIntensityAnalyzer()
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        feature_list = []
        for text in X:
            clean = clean_text(text)
            words = re.findall(r'\w+', clean)
            lower_words = [word.lower() for word in words]
            
            # Basic handcrafted features.
            swear_count = sum(1 for word in lower_words if word in self.swear_words)
            hate_speech_count = sum(1 for word in lower_words if word in self.hate_speech_words)
            extremist_count = sum(1 for word in lower_words if word in self.extremist_words)
            all_caps_count = sum(1 for word in re.findall(r'\b[A-Z]{2,}\b', clean))
            exclamation_count = clean.count('!')
            question_count = clean.count('?')
            word_count = len(words)
            letters = re.findall(r'[A-Za-z]', clean)
            capital_ratio = (sum(1 for letter in letters if letter.isupper()) / len(letters)) if letters else 0.0
            avg_word_length = np.mean([len(word) for word in words]) if words else 0.0
            punctuation_ratio = (exclamation_count + question_count) / word_count if word_count > 0 else 0.0
            elongated_words = re.findall(r'\b\w*(\w)\1{2,}\w*\b', text)
            elongated_word_count = len(elongated_words)
            
            # MWE count feature.
            mwe_count = 0
            for mwe in self.mwe_list:
                pattern = r'\b' + r'\s+'.join(mwe) + r'\b'
                matches = re.findall(pattern, clean, flags=re.IGNORECASE)
                mwe_count += len(matches)
            
            # Sentiment features using VADER.
            sent_scores = self.sia.polarity_scores(text)
            compound_sent = sent_scores['compound']
            pos_sent = sent_scores['pos']
            neu_sent = sent_scores['neu']
            neg_sent = sent_scores['neg']
            
            features = [
                swear_count,
                hate_speech_count,
                extremist_count,
                all_caps_count,
                exclamation_count,
                question_count,
                word_count,
                capital_ratio,
                avg_word_length,
                punctuation_ratio,
                elongated_word_count,
                mwe_count,
                compound_sent,
                pos_sent,
                neu_sent,
                neg_sent
            ]
            feature_list.append(features)
        return np.array(feature_list)

# -----------------
# Main Execution
# -----------------

# Load the dataset.
df = pd.read_csv('train.csv')
df = df[['tweet', 'class']].dropna()

# Extract candidate MWEs from the corpus.
corpus = df['tweet'].tolist()
mwe_extracted = extract_mwes_from_corpus(corpus, top_n=50, freq_filter=3)
print("Extracted MWEs:", mwe_extracted)

# Combine handcrafted and TF-IDF features.
combined_features = FeatureUnion([
    ('handcrafted', IncendiaryContentFeatureExtractor(mwe_list=mwe_extracted)),
    ('tfidf', TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), max_features=300))
])

# Build pipeline with resampling and updated Logistic Regression.
pipeline = ImbPipeline([
    ('features', combined_features),
    ('scaler', StandardScaler(with_mean=False)),
    ('resample', SMOTETomek(random_state=42)),
    ('clf', LogisticRegression(
        max_iter=3000,
        random_state=42,
        solver='saga'
    ))
], memory=memory)

# Split the dataset.
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['class'], test_size=0.2, random_state=42)

# Set up grid search: use fewer CV folds and a smaller parameter grid.
param_grid = {
    'clf__C': [0.1, 1, 10],
    'clf__class_weight': [None, 'balanced']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted', n_jobs=1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

# Evaluate the final model.
y_pred = grid_search.predict(X_test)
print("Enhanced Pipeline with Resampling, Hybrid Features, and Sentiment Scores (Optimized Version)")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Weighted F1 Score: {:.4f}".format(f1_score(y_test, y_pred, average='weighted')))
print("Macro F1 Score: {:.4f}".format(f1_score(y_test, y_pred, average='macro')))
