import matplotlib.pyplot as plt
import numpy as np

# 新的模型名称和每类的 F1-Score（0: Hate Speech, 1: Offensive Language, 2: Neither）
models = ['Logistic Regression', 'Naive Bayes', 'XLNet', 'BERTweet-base', 'BERTweet-large']
f1_class_0 = [0.33, 0.19, 0.44, 0.47, 0.47]
f1_class_1 = [0.90, 0.92, 0.95, 0.95, 0.95]
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
plt.show()
