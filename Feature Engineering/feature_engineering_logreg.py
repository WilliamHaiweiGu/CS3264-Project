import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
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
    return finder.nbest(bigram_measures.pmi, top_n)

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
        self.mwe_list = mwe_list or []
        self.sia = SentimentIntensityAnalyzer()
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        feature_list = []
        for text in X:
            clean = clean_text(text)
            words = re.findall(r'\w+', clean)
            lower_words = [w.lower() for w in words]
            # handcrafted features
            swear_count = sum(w in self.swear_words for w in lower_words)
            hate_count = sum(w in self.hate_speech_words for w in lower_words)
            ext_count = sum(w in self.extremist_words for w in lower_words)
            all_caps = len(re.findall(r'\b[A-Z]{2,}\b', clean))
            exclamation = clean.count('!')
            question = clean.count('?')
            word_count = len(words)
            letters = re.findall(r'[A-Za-z]', clean)
            cap_ratio = sum(l.isupper() for l in letters)/len(letters) if letters else 0.0
            avg_len = np.mean([len(w) for w in words]) if words else 0.0
            punct_ratio = (exclamation + question)/word_count if word_count else 0.0
            elongated = len(re.findall(r'\b\w*(\w)\1{2,}\w*\b', text))
            mwe_count = sum(len(re.findall(r'\b' + r'\s+'.join(mwe) + r'\b', clean, flags=re.IGNORECASE))
                            for mwe in self.mwe_list)
            # sentiment
            sent = self.sia.polarity_scores(text)
            features = [
                swear_count, hate_count, ext_count, all_caps,
                exclamation, question, word_count, cap_ratio,
                avg_len, punct_ratio, elongated, mwe_count,
                sent['compound'], sent['pos'], sent['neu'], sent['neg']
            ]
            feature_list.append(features)
        return np.array(feature_list)

# Main Execution
df = pd.read_csv('train.csv')[['tweet', 'class']].dropna()
corpus = df['tweet'].tolist()
mwe_list = extract_mwes_from_corpus(corpus, top_n=50, freq_filter=3)
print("Extracted MWEs:", mwe_list)

combined = FeatureUnion([
    ('handcrafted', IncendiaryContentFeatureExtractor(mwe_list=mwe_list)),
    ('tfidf', TfidfVectorizer(preprocessor=clean_text, ngram_range=(1,2), max_features=300))
])
pipeline = ImbPipeline([
    ('features', combined),
    ('scaler', StandardScaler(with_mean=False)),
    ('resample', SMOTETomek(random_state=42)),
    ('clf', LogisticRegression(max_iter=3000, random_state=42, solver='saga'))
], memory=memory)

X_train, X_test, y_train, y_test = train_test_split(
    df['tweet'], df['class'], test_size=0.2, random_state=42
)
param_grid = {'clf__C':[0.1,1,10], 'clf__class_weight':[None,'balanced']}
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted', n_jobs=1)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)

# Predict and evaluate
y_pred = grid.predict(X_test)
report_dict = classification_report(y_test, y_pred, output_dict=True)
ma = report_dict['macro avg']
wa = report_dict['weighted avg']
accuracy = accuracy_score(y_test, y_pred)

# Print requested table rows
print(f"{'Class':<30}{'Precision':>10}{'Recall':>10}{'F1-Score':>12}")
for cls in ['0','1','2']:
    name = {
        '0':'0 (Hate Speech)',
        '1':'1 (Offensive Language)',
        '2':'2 (Neither)'
    }[cls]
    stats = report_dict[cls]
    print(f"{name:<30}{stats['precision']:.2f}{stats['recall']:>10.2f}{stats['f1-score']:>12.2f}")

# Average F1 only
print(f"{'Average':<30}{'–':>10}{'–':>10}{report_dict['macro avg']['f1-score']:>12.2f}")
# Macro Avg with precision and recall
print(f"{'Macro Avg':<30}{ma['precision']:.2f}{ma['recall']:>10.2f}{ma['f1-score']:>12.2f}")
# Weighted Avg with precision and recall
print(f"{'Weighted Avg':<30}{wa['precision']:.2f}{wa['recall']:>10.2f}{wa['f1-score']:>12.2f}")