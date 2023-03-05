import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 데이터 불러오기
data = pd.read_csv('./dataset/inflearn_creditcard.csv')

# features_labels 분리
features = data.drop(['Class'], axis=1).values
labels = np.array(data.pop('Class'))


# train_test_split
train, test = train_test_split(features, labels, test_size=0.5)