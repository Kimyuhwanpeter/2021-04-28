# 2021-04-28
* original GEI 모델에 문제가 생겨서 다시 고쳤음 (concat 문제가 있었음)
* focal loss 적용 (오직 feature vector 출력, model_V16)
* model_V17은 두 feature vector 출력에서 focal loss 적용
* V18은 학습할 때 각 class에 해당하는 이미지들의 대표 이미지를 입력으로 하나 더 추가하였음
<br/>

* 세 모델 다 코랩에 일단 돌려볼것
