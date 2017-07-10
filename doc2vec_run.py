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


# 실제 구동 데이터를 읽기
run_data = read_data('data/ratings_run.txt')

# 형태소 분류
run_docs = [(tokenize(row[1]), row[2]) for row in run_data[1:]]

# doc2vec 에서 필요한 데이터 형식으로 변경
TaggedDocument = namedtuple('TaggedDocument', 'words tags')
tagged_run_docs = [TaggedDocument(d, [c]) for d, c in run_docs]

# load train data
doc_vectorizer = Doc2Vec.load('model/doc2vec.model')

# 분류를 위한 피쳐 생성
run_x = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_run_docs]
run_y = [doc.tags[0] for doc in tagged_run_docs]

# load the model from disk
filename = 'model/finalized_model.sav'

# 실제 분류 확인
loaded_model = pickle.load(open(filename, 'rb'))
print(loaded_model.predict(run_x[0].reshape(1, -1)))
print(loaded_model.predict(run_x[1].reshape(1, -1)))
