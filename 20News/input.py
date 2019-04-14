# 训练和测试数据的提取
from sklearn.datasets import fetch_20newsgroups

# 训练集和测试集
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
train_texts = newsgroups_train['data']
train_labels = newsgroups_train['target']
test_texts = newsgroups_test['data']
test_labels = newsgroups_test['target']
