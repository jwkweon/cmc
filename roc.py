import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# 예시 데이터
y_true = np.array([0, 2, 1, 2, 2, 0, 1, 0])
y_scores = np.array([
    [0.2, 0.5, 0.3, 0.0, 0.0, 0.0],
    [0.1, 0.3, 0.6, 0.0, 0.0, 0.0],
    [0.3, 0.2, 0.5, 0.0, 0.0, 0.0],
    [0.2, 0.2, 0.6, 0.0, 0.0, 0.0],
    [0.4, 0.3, 0.3, 0.0, 0.0, 0.0],
    [0.6, 0.3, 0.1, 0.0, 0.0, 0.0],
    [0.4, 0.5, 0.1, 0.0, 0.0, 0.0],
    [0.7, 0.2, 0.1, 0.0, 0.0, 0.0]
])

n_classes = 6
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5])

fpr = dict()
tpr = dict()
roc_auc = dict()

# 각 클래스에 대한 ROC 커브 계산
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# ROC 커브 플롯
colors = ['#D5A7E3', '#A8E6CF', '#DCEDC1', '#FFD3B6', '#FFAAA5', '#FF8B94']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Multi-Class')
plt.legend(loc="lower right")
plt.show()
