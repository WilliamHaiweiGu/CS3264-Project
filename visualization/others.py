import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

# ==== Data for Confusion Matrix (BERTweet-large) ====
# Let's assume we have the following true vs predicted labels (dummy data for illustration)
true_labels = [0, 1, 1, 2, 0, 2, 1, 1, 2, 0, 2, 1, 1, 1, 2, 0, 2, 2, 1, 2]
pred_labels = [0, 1, 1, 2, 1, 2, 1, 1, 2, 0, 2, 1, 1, 1, 1, 0, 2, 2, 1, 2]

# ==== Confusion Matrix Heatmap ====
conf_mat = confusion_matrix(true_labels, pred_labels, labels=[0,1,2])
conf_df = pd.DataFrame(conf_mat, index=["Hate Speech", "Offensive", "Neither"],
                                  columns=["Hate Speech", "Offensive", "Neither"])

plt.figure(figsize=(6, 5))
sns.heatmap(conf_df, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - BERTweet-large")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()

# ==== Radar Chart (Macro Avg) ====
import matplotlib.pyplot as plt
from math import pi

# Data
models = ['LogReg', 'Naive Bayes', 'XLNet', 'BERTweet-base', 'BERTweet-large']
macro_precision = [0.64, 0.79, 0.76, 0.78, 0.79]
macro_recall = [0.74, 0.56, 0.76, 0.77, 0.76]
macro_f1 = [0.67, 0.60, 0.76, 0.78, 0.78]

# Create DataFrame for radar chart
df = pd.DataFrame({
    'model': models,
    'Precision': macro_precision,
    'Recall': macro_recall,
    'F1': macro_f1
})

# Radar chart settings
categories = list(df.columns[1:])
num_vars = len(categories)

angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # complete the loop

plt.figure(figsize=(8, 8))

for i in range(len(df)):
    values = df.iloc[i].drop('model').tolist()
    values += values[:1]  # complete the loop
    plt.polar(angles, values, label=df.iloc[i]['model'], linewidth=2)

plt.xticks(angles[:-1], categories)
plt.title('Macro Metrics Radar Chart')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.show()

# ==== Precision-Recall-F1 Bar Chart for Each Class (BERTweet-large) ====
classes = ['Hate Speech', 'Offensive Language', 'Neither']
metrics = ['Precision', 'Recall', 'F1']
values = [
    [0.49, 0.44, 0.47],  # Hate Speech
    [0.94, 0.96, 0.95],  # Offensive
    [0.94, 0.88, 0.91],  # Neither
]

x = np.arange(len(metrics))
width = 0.2

fig, ax = plt.subplots(figsize=(8, 6))
for i, (label, v) in enumerate(zip(classes, values)):
    ax.bar(x + i*width, v, width, label=label)

ax.set_ylabel('Score')
ax.set_title('Precision, Recall, F1 - BERTweet-large')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.set_ylim([0, 1])
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
