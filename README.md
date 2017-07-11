# classify_comment_emotion
PyCon Korea 2015에서 발표된 [한국어와 NLTK, Gensim](https://www.lucypark.kr/slides/2015-pyconkr/#1)을 시도해 볼 수 있는 샘플코드입니다. 저는 친철한; 발표자료를 보고도 따라해보는게 어려웠어서, 하나씩 테스트해 볼 수 있도록 하였습니다.

## 요구사항
- python 3
- doc2vec
- konlpy
- gensim
- sklearn
- numpy
- pickle

## 실행방법
- ### 문장을 형태소 분석하여 NaiveBayesClassifier을 통해 긍정,부정을 검증
  - python run_nltk.py

- ### doc2vec를 활용하여 Gensim을 통해 긍정, 부정 분류
  - 리뷰 데이터 학습
    - python doc2vec_train.py
  - 학습된 데이터 검증
    - python doc2vec_test.py
  - 검증까지 완료된 모델을 통해 실제 리뷰로 테스트
    - python doc2vec_run.py    
