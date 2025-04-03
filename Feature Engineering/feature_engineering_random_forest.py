import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score
from imblearn.pipeline import Pipeline  # imblearn's Pipeline supports resampling
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier

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
        # Use frozensets for immutability
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
            clean = clean_text(text)
            words = re.findall(r'\w+', clean)
            lower_words = [word.lower() for word in words]
            
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

# Load data
df = pd.read_csv('train.csv')
df = df[['tweet', 'class']].dropna()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['class'], test_size=0.2, random_state=42)

# Build a new pipeline using RandomForestClassifier
pipeline_rf = Pipeline([
    ('features', IncendiaryContentFeatureExtractor()),
    ('scaler', StandardScaler()),
    ('adasyn', ADASYN(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])

param_grid_rf = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20],
    'clf__class_weight': [None, 'balanced']
}

grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='f1_weighted')
grid_search_rf.fit(X_train, y_train)

print("Best parameters for RandomForest:", grid_search_rf.best_params_)

y_pred_rf = grid_search_rf.predict(X_test)
print("Incendiary Content Feature Engineering with RandomForest and ADASYN (Grid Search)")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Weighted F1 Score: {:.4f}".format(f1_score(y_test, y_pred_rf, average='weighted')))
