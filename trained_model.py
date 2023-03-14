'''
[You Don’t Need Neural Networks to Do Continual Learning] https://towardsdatascience.com/you-dont-need-neural-networks-to-do-continual-learning-2ed3bfe3dbfc
[TabNet: Attentive Interpretable Tabular Learning] https://openreview.net/forum?id=BylRkAEKDH
[Self-Supervised Learning on Tabular Data with TabNet] https://medium.com/@vanillaxiangshuyang/self-supervised-learning-on-tabular-data-with-tabnet-544b3ec85cee
[[Review] TABNET: Attentive Interpretable Tabular Learning (2019)] https://wsshin.tistory.com/5
'''

import pickle
import numpy as np

from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score

# load data
with open( "creditcard", "rb" ) as file:
    data = pickle.load(file)

features = data.drop(['Class'], axis=1).values
labels = np.array(data.pop('Class'))

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, stratify=labels, random_state=42)

print("전체 data의 positive 건수 : ", Counter(labels))
print("Train set 의 positive 건수 : ", Counter(y_train))
print("Test set 의 positive 건수 : ", Counter(y_test))

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# model
'''
n_estimators : 모델에서 사용할 트리 갯수(학습시 생성할 트리 갯수)
criterion : 분할 품질을 측정하는 기능 (default : gini)
max_depth : 트리의 최대 깊이
min_samples_split : 내부 노드를 분할하는데 필요한 최소 샘플 수 (default : 2)
min_samples_leaf : 리프 노드에 있어야 할 최소 샘플 수 (default : 1)
min_weight_fraction_leaf : min_sample_leaf와 같지만 가중치가 부여된 샘플 수에서의 비율
max_features : 각 노드에서 분할에 사용할 특징의 최대 수
max_leaf_nodes : 리프 노드의 최대수
min_impurity_decrease : 최소 불순도
min_impurity_split : 나무 성장을 멈추기 위한 임계치
bootstrap : 부트스트랩(중복허용 샘플링) 사용 여부
oob_score : 일반화 정확도를 줄이기 위해 밖의 샘플 사용 여부
n_jobs :적합성과 예측성을 위해 병렬로 실행할 작업 수
random_state : 난수 seed 설정
verbose : 실행 과정 출력 여부
warm_start : 이전 호출의 솔루션을 재사용하여 합계에 더 많은 견적가를 추가
class_weight : 클래스 가중치
'''
model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:, 1]

y_pred = y_pred_prob > 0.5

print("Test set의 positive 건수 = ", sum(y_test))
print("Prediction의 positive  건수 = ", sum(y_pred))
print("accuracy = {:.5f}".format(sum(y_pred == y_test) / len(y_test)))

# 평가
def plot_cm(y_test, y_pred_proba, threshold):
    
    y_pred = y_pred_proba > threshold
    
    cm = confusion_matrix(y_test, y_pred)
    
    print("f1 score:", f1_score(y_test, y_pred))
    print("Accuracy", accuracy_score(y_test, y_pred))
    print("Precision", precision_score(y_test, y_pred))
    print("Recall", recall_score(y_test, y_pred))
    
    plt.figure(figsize=(5,5))

    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix (threshold>{:.2f}) '.format(threshold))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
plot_cm(y_test, y_pred_prob, 0.5)