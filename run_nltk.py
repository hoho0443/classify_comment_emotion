from konlpy.tag import Twitter
import nltk

twitter = Twitter()

print(twitter.morphs(u'한글형태소분석기 테스트 중 입니다')) # ??
print(twitter.nouns(u'한글형태소분석기 테스트 중 입니다!')) #명사
print(twitter.pos(u'한글형태소분석기 테스트 중 입니다.')) #형태소

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

def tokenize(doc):
  # norm, stem은 optional
  return ['/'.join(t) for t in twitter.pos(doc, norm=True, stem=True)]

def term_exists(doc):
    return {'exists({})'.format(word): (word in set(doc)) for word in selected_words}

# 트래이닝 데이터와 테스트 데이터를 읽기
train_data = read_data('data/ratings_train.txt')
test_data = read_data('data/ratings_test.txt')

# row, column의 수가 제대로 읽혔는지 확인
print(len(train_data))      # nrows: 150000
print(len(train_data[0]))   # ncols: 3
print(len(test_data))       # nrows: 50000
print(len(test_data[0]))     # ncols: 3

# 형태소 분류
train_docs = [(tokenize(row[1]), row[2]) for row in train_data[1:]]
test_docs = [(tokenize(row[1]), row[2]) for row in test_data[1:]]

#Training data의 token 모으기
tokens = [t for d in train_docs for t in d[0]]
print(len(tokens))

# Load tokens with nltk.Text()
text = nltk.Text(tokens, name='NMSC')
print(text.vocab().most_common(10))

# 텍스트간의 연어 빈번하게 등장하는 단어 구하기
text.collocations()


# term이 존재하는지에 따라서 문서를 분류
selected_words = [f[0] for f in text.vocab().most_common(2000)] # 여기서는 최빈도 단어 2000개를 피쳐로 사용
train_docs = train_docs[:10000] # 시간 단축을 위한 꼼수로 training corpus의 일부만 사용할 수 있음
train_xy = [(term_exists(d), c) for d, c in train_docs]
test_xy = [(term_exists(d), c) for d, c in test_docs]

# nltk의 NaiveBayesClassifier으로 데이터를 트래이닝 시키고, test 데이터로 확인
classifier = nltk.NaiveBayesClassifier.train(train_xy) #Naive Bayes classifier 적용
print(nltk.classify.accuracy(classifier, test_xy))
# => 0.80418

classifier.show_most_informative_features(10)







#nltk.polarity_scores("i love you")
