from collections import namedtuple
from gensim.models import doc2vec
from konlpy.tag import Twitter
import multiprocessing
from pprint import pprint

twitter = Twitter()

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

def tokenize(doc):
  # norm, stem은 optional
  return ['/'.join(t) for t in twitter.pos(doc, norm=True, stem=True)]



#doc2vec parameters
cores = multiprocessing.cpu_count()

vector_size = 300
window_size = 15
word_min_count = 2
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 1
worker_count = cores


# 트래이닝 데이터 읽기
train_data = read_data('data/ratings_train.txt')

# 형태소 분류
train_docs = [(tokenize(row[1]), row[2]) for row in train_data[1:]]

# doc2vec 에서 필요한 데이터 형식으로 변경
TaggedDocument = namedtuple('TaggedDocument', 'words tags')
tagged_train_docs = [TaggedDocument(d, [c]) for d, c in train_docs]

# 사전 구축
doc_vectorizer = doc2vec.Doc2Vec(size=300, alpha=0.025, min_alpha=0.025, seed=1234)
doc_vectorizer.build_vocab(tagged_train_docs)

# Train document vectors!
for epoch in range(10):
    doc_vectorizer.train(tagged_train_docs, total_examples=doc_vectorizer.corpus_count, epochs=doc_vectorizer.iter)
    doc_vectorizer.alpha -= 0.002  # decrease the learning rate
    doc_vectorizer.min_alpha = doc_vectorizer.alpha  # fix the learning rate, no decay

#To save
doc_vectorizer.save('model/doc2vec.model')

pprint(doc_vectorizer.most_similar('공포/Noun'))
pprint(doc_vectorizer.similarity('공포/Noun', 'ㅋㅋ/KoreanParticle'))
