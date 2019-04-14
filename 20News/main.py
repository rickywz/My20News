import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from input import *
from sklearn.metrics import classification_report

text_clf = Pipeline([('tfidf', TfidfVectorizer(max_features=10000)), ('clf', GradientBoostingClassifier())])
text_clf = text_clf.fit(train_texts, train_labels)
predicted = text_clf.predict(test_texts)
print("GradientBoostingClassifier准确率为：", np.mean(predicted == test_labels))
print(classification_report(newsgroups_test.target, predicted, target_names= ['alt.atheism',
                                                                             'comp.graphics',
                                                                             'comp.os.ms-windows.misc',
                                                                             'comp.sys.ibm.pc.hardware',
                                                                             'comp.sys.mac.hardware',
                                                                             'comp.windows.x',
                                                                             'misc.forsale',
                                                                             'rec.autos',
                                                                             'rec.motorcycles',
                                                                             'rec.sport.baseball',
                                                                             'rec.sport.hockey',
                                                                             'sci.crypt',
                                                                             'sci.electronics',
                                                                             'sci.med',
                                                                             'sci.space',
                                                                             'soc.religion.christian',
                                                                             'talk.politics.guns',
                                                                             'talk.politics.mideast',
                                                                             'talk.politics.misc',
                                                                             'talk.religion.misc']))
