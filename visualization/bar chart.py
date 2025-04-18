import matplotlib.pyplot as plt
import numpy as np

# 新的模型名称和每类的 F1-Score（0: Hate Speech, 1: Offensive Language, 2: Neither）
models = ['Logistic Regression', 'Naive Bayes', 'XLNet', 'BERTweet-base', 'BERTweet-large']
f1_class_0 = [0.32, 0.15, 0.44, 0.46, 0.45]
f1_class_1 = [0.89, 0.92, 0.95, 0.95, 0.95]
f1_class_2 = [0.79, 0.73, 0.89, 0.91, 0.91]

x = np.arange(len(models))
width = 0.25

# 设置颜色
colors = ['#4C72B0', '#55A868', '#C44E52']

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, f1_class_0, width, label='Hate Speech', color=colors[0])
ax.bar(x, f1_class_1, width, label='Offensive Language', color=colors[1])
ax.bar(x + width, f1_class_2, width, label='Neither', color=colors[2])

ax.set_ylabel('F1 Score')
ax.set_title('F1 Score per Class by Model')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.set_ylim([0, 1])
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
# plt.show()

macro_f1 = [0.67, 0.61, 0.76, 0.77, 0.77]
weighted_f1 = [0.84, 0.85, 0.76, 0.91, 0.91]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, macro_f1, width, label='Macro F1', color='#2b83ba')
ax.bar(x + width/2, weighted_f1, width, label='Weighted F1', color='#fdae61')

ax.set_ylabel('F1 Score')
ax.set_title('Macro and Weighted F1 Scores by Model')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.set_ylim([0, 1])
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
