import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def clean_text(text: str) -> str:
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^A-Za-z.!?'0-9]", ' ', text)
    text = re.sub(r'\t', ' ', text)
    return re.sub(r" +", ' ', text)

class IncendiaryContentFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract handcrafted features indicative of:
    - Hate Speech (e.g., use of hateful slurs)
    - Rage-Baiting (e.g., emotional punctuation, all-caps words)
    - Self-Radicalization (e.g., extremist ideology keywords)
    along with general textual features.
    """
    def __init__(self, 
                 swear_words=None, 
                 hate_speech_words=None, 
                 extremist_words=None):
        # Convert mutable sets to immutable frozensets
        self.swear_words = frozenset(swear_words) if swear_words else frozenset({
            'fuck', 'shit', 'damn', 'bitch', 'asshole', 'crap', 
            'dick', 'piss', 'cunt', 'cock', 'prick', 'bastard', 'slut', 'whore'
        })
        self.hate_speech_words = frozenset(hate_speech_words) if hate_speech_words else frozenset({
            'nigger', 'chink', 'spic', 'kike', 'faggot', 'dyke', 'gook', 
            'wetback', 'towelhead', 'camel jockey'
        })
        self.extremist_words = frozenset(extremist_words) if extremist_words else frozenset({
            'jihad', 'isis', 'al-qaeda', 'extremist', 'martyr', 'revolution', 'uprising', 
            'caliphate', 'sharia', 'takfir', 'khilafah', 'jihadi', 'terrorist', 'suicide bomber'
        })
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        feature_list = []
        for text in X:
            # Clean text using our improved clean_text function
            clean = clean_text(text)
            
            # Extract words for dictionary-based counts
            words = re.findall(r'\w+', clean)
            lower_words = [word.lower() for word in words]
            
            # Feature 1: Swear word count
            swear_count = sum(1 for word in lower_words if word in self.swear_words)
            
            # Feature 2: Hate speech word count
            hate_speech_count = sum(1 for word in lower_words if word in self.hate_speech_words)
            
            # Feature 3: Extremist keyword count (for self-radicalization cues)
            extremist_count = sum(1 for word in lower_words if word in self.extremist_words)
            
            # Feature 4: Count of all-caps words (at least 2 letters long)
            all_caps_count = sum(1 for word in re.findall(r'\b[A-Z]{2,}\b', clean))
            
            # Feature 5: Count of exclamation marks (emotional intensity)
            exclamation_count = clean.count('!')
            
            # Feature 6: Count of question marks (rhetorical or provocative questioning)
            question_count = clean.count('?')
            
            # Feature 7: Total number of words
            word_count = len(words)
            
            # Feature 8: Ratio of capital letters to total letters
            letters = re.findall(r'[A-Za-z]', clean)
            capital_ratio = (sum(1 for letter in letters if letter.isupper()) / len(letters)) if letters else 0.0
            
            # Feature 9: Average word length
            avg_word_length = np.mean([len(word) for word in words]) if words else 0.0

            feature_list.append([
                swear_count,
                hate_speech_count,
                extremist_count,
                all_caps_count,
                exclamation_count,
                question_count,
                word_count,
                capital_ratio,
                avg_word_length
            ])
        return np.array(feature_list)

# Load data and select relevant columns
df = pd.read_csv('train.csv')
df = df[['tweet', 'class']].dropna()

# Build a pipeline: custom feature extraction, scaling, and logistic regression
pipeline = Pipeline([
    ('features', IncendiaryContentFeatureExtractor()),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['class'], test_size=0.2, random_state=42)

# Set up the grid search to tune Logistic Regression parameters
param_grid = {
    'clf__C': [0.01, 0.1, 1, 10, 100],
    'clf__class_weight': [None, 'balanced']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

# Use the best estimator from grid search to predict and evaluate performance
y_pred = grid_search.predict(X_test)
print("Incendiary Content Feature Engineering with Logistic Regression (Grid Search)")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Weighted F1 Score: {:.4f}".format(f1_score(y_test, y_pred, average='weighted')))
