from collections import namedtuple
from gensim.models import doc2vec
from konlpy.tag import Twitter
import multiprocessing
from pprint import pprint
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression
import numpy
import pickle


twitter = Twitter()

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

def tokenize(doc):
  # norm, stem은 optional
  return ['/'.join(t) for t in twitter.pos(doc, norm=True, stem=True)]


# 테스트 데이터를 읽기
train_data = read_data('data/ratings_train.txt')
test_data = read_data('data/ratings_test.txt')

# 형태소 분류
train_docs = [(tokenize(row[1]), row[2]) for row in train_data[1:]]
test_docs = [(tokenize(row[1]), row[2]) for row in test_data[1:]]

# doc2vec 에서 필요한 데이터 형식으로 변경
TaggedDocument = namedtuple('TaggedDocument', 'words tags')
tagged_train_docs = [TaggedDocument(d, [c]) for d, c in train_docs]
tagged_test_docs = [TaggedDocument(d, [c]) for d, c in test_docs]

# load train data
doc_vectorizer = Doc2Vec.load('model/doc2vec.model')

# 분류를 위한 피쳐 생성
train_x = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_train_docs]
train_y = [doc.tags[0] for doc in tagged_train_docs]
test_x = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_test_docs]
test_y = [doc.tags[0] for doc in tagged_test_docs]


#classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
classifier = LogisticRegression(random_state=1234)
classifier.fit(train_x, train_y)

# 테스트 socre 확인
print( classifier.score(test_x, test_y) )
# 0.63904

# save the model to disk
filename = 'model/finalized_model.sav'
pickle.dump(classifier, open(filename, 'wb'))
