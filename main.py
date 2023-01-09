# -*- coding: utf-8 -*-

## [Dacon] 블럭 장난감 제조 공정 최적화 경진대회
## (팀명)
## 2020년 월 일 (제출날짜)

## 1. 라이브러리 및 데이터
## Library & Data
import pandas as pd
import numpy as np
import multiprocessing
import warnings
from copy import deepcopy
from module.genome import Genome, genome_score

from module import __version__

warnings.filterwarnings(action='ignore')
np.random.seed(777)

print('Pandas : %s' % (pd.__version__))
print('Numpy : %s' % (np.__version__))

## 2. 데이터 전처리
## Data Cleansing & Pre-Processing

## 3. 탐색적 자료분석
## Exploratory Data Analysis

## 4. 변수 선택 및 모델 구축
## Feature Engineering & Initial Modeling


def save_submission_file(best_genomes, output_name):
    from module.simulator import Simulator
    # simulator = Simulator()
    # order = pd.read_csv('module/order.csv')
    submission = best_genomes[0].submission
    df_stock = best_genomes[0].df_stock
    #submission, df_stock = best_genomes[0].predict(order)
    # _, df_stock = best_genomes[0].get_score(submission)
    # PRT 개수 계산
    # The maximum required PRT per hour is 11 (5.5 * 2 lines)
    # The maximum required PRT per day is 11 * 24 < 500, which is the maximum number of PRT inputs
    # Therefore, we don't have to worry about the NG case for this 'reversed' PRT calculation.
    PRTs = df_stock[['PRT_1', 'PRT_2', 'PRT_3', 'PRT_4']].values
    PRTs = (PRTs[:-1] - PRTs[1:])[24 * 23:]
    PRTs = np.ceil(PRTs * 1.1)
    PAD = np.zeros((24 * 23 + 1, 4))
    PRTs = np.append(PRTs, PAD, axis=0).astype(int)
    # Submission 파일에 PRT 입력
    submission.loc[:, 'PRT_1':'PRT_4'] = PRTs
    submission.to_csv(output_name, index=False)


CPU_CORE = multiprocessing.cpu_count()  # 멀티프로세싱 CPU 사용 수
N_POPULATION = 30  # 세대당 생성수
N_BEST = 3  # 베스트 수
N_CHILDREN = 3  # 자손 유전자 수
PROB_MUTATION = 0.5  # 돌연변이
REVERSE = True  # 배열 순서 (False: ascending order, True: descending order)

score_ini = 10  # 초기 점수
score_lower_bound_for_new_half = 30  # 점수 보다 낮으면 절반은 새로 생성
input_length = 40 + 4  # 입력 데이터 길이 (10일치 blk 수요 + 현재 blk 재고)
output_length_1 = 18  # Event (CHECK_1~4, CHANGE_12~CHANGE_43, PROCESS, STOP)
output_length_2 = 12  # MOL (0~5.5, step:0.5)
h1 = 50  # 히든레이어1 노드 수
h2 = 50  # 히든레이어2 노드 수
h3 = 50  # 히든레이어3 노드 수
EPOCHS = 50

EARLY_SAVE_MIN_SCORE = 90
USE_PRETRAINED_GENOMES = False
PRETRAINED_BEST_SCORE = 70.54105
MODEL_VERSION = __version__

ORDER_FILE = None
ORDER_FILE = 'module/order_XXX.csv'  # Change order file to learn other cases

import logging
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%m_%d_%H:%M")
log_file = "log_popn{}_h1{}_h2{}_h3{}_epochs{}_{}.log".format(N_POPULATION, h1, h2, h3, EPOCHS, dt_string)
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.info("#Parameters:\tN_POP:{}\tN_BEST:{}\tN_CHILDREN:{}\tPROB_MUTATION:{}\tREVERSE:{}".format(N_POPULATION, N_BEST, N_CHILDREN, PROB_MUTATION, REVERSE))
logging.info("#Parameters:\tINPUT_LENGTH:{}\tOUTPUT_LENGTH1:{}\tOUTPUT_LENGTH2:{}\th1:{}\th2:{}\th3:{}\tEPOCHS:{}".format(input_length, output_length_1, output_length_2, h1, h2, h3, EPOCHS))
logging.info("CHECK, PROCESS, STOP, and CHANGE are all considered")

genomes = []
for _ in range(N_POPULATION):
    genome = Genome(score_ini, input_length, output_length_1, output_length_2, h1, h2, h3)
    genomes.append(genome)

if USE_PRETRAINED_GENOMES:
    assert PRETRAINED_BEST_SCORE != 0
    logging.info("Used pre-trained models! best score: {}, N_BEST: {}".format(PRETRAINED_BEST_SCORE, N_BEST))
    # Load best genomes
    import pickle
    best_genomes = []
    for i in range(N_BEST):
        with open("{}v_best_genomes_{:.5f}_{}.pkl".format(MODEL_VERSION, PRETRAINED_BEST_SCORE, i), "rb") as input:
            if ORDER_FILE:
                best_genome = pickle.load(input)
                best_genome.change_order_file(ORDER_FILE)
                genome_score(best_genome)
                best_genomes.append(best_genome)
                logging.info("Prev score:{} -> Updated score with new order file:{}".format(PRETRAINED_BEST_SCORE, best_genome.score))
            else:
                best_genomes.append(pickle.load(input))
            print("Loaded {}th best genome, score: {}".format(i, best_genomes[i].score))

    for i in range(N_BEST):
        genome_score(best_genomes[i])  # Best genome score update (There might be a change in the predict and mask code)
        genomes[i] = best_genomes[i]
else:
    best_genomes = []
    for _ in range(N_BEST):
        genome = Genome(score_ini, input_length, output_length_1, output_length_2, h1, h2, h3)
        best_genomes.append(genome)

## 5. 모델 학습 및 검증
## Model Tuning & Evaluation
n_gen = 1
score_history = []
high_score_history = []
mean_score_history = []
while n_gen <= EPOCHS:
    genome_score(genomes[0])
    genomes = np.array(genomes)
    while len(genomes) % CPU_CORE != 0:
        genomes = np.append(genomes, Genome(score_ini, input_length, output_length_1, output_length_2, h1, h2, h3))
    genomes = genomes.reshape((len(genomes) // CPU_CORE, CPU_CORE))

    for idx, _genomes in enumerate(genomes):
        if __name__ == '__main__':
            pool = multiprocessing.Pool(processes=CPU_CORE)
            genomes[idx] = pool.map(genome_score, _genomes)
            pool.close()
            pool.join()
    genomes = list(genomes.reshape(genomes.shape[0] * genomes.shape[1]))

    # score에 따라 정렬
    genomes.sort(key=lambda x: x.score, reverse=REVERSE)

    # 평균 점수
    s = 0
    for i in range(N_BEST):
        s += genomes[i].score
    s /= N_BEST

    # Best Score
    bs = genomes[0].score

    # Best Model 추가
    if best_genomes is not None:
        genomes.extend(best_genomes)

    # score에 따라 정렬
    genomes.sort(key=lambda x: x.score, reverse=REVERSE)

    score_history.append([n_gen, genomes[0].score])
    high_score_history.append([n_gen, bs])
    mean_score_history.append([n_gen, s])

    # 결과 출력
    current_time = datetime.now()
    print('%s EPOCH #%s\tHistory Best Score: %s\tBest Score: %s\tMean Score: %s' % (current_time, n_gen, genomes[0].score, bs, s))
    logging.info('EPOCH #%s\tHistory Best Score: %s\tBest Score: %s\tMean Score: %s' % (n_gen, genomes[0].score, bs, s))

    # 모델 업데이트
    best_genomes = deepcopy(genomes[:N_BEST])

    # Early stop
    # if n_gen > 5 and bs > 80:
    #     print("Score difference {}".format(abs(mean_score_history[-2][1] - mean_score_history[-1][1])))
    #     if abs(mean_score_history[-2][1] - mean_score_history[-1][1]) < 0.00005:
    #         logging.info("Early stop occurred!")
    #         break

    if bs > EARLY_SAVE_MIN_SCORE:
        logging.info("Save submission to {}".format('{}v_early_best_gen{}_score{:.5f}.csv'.format(__version__, n_gen, bs)))
        save_submission_file(best_genomes, '{}v_early_best_gen{}_score{:.5f}.csv'.format(__version__, n_gen, bs))

    # CHILDREN 생성
    for i in range(N_CHILDREN):
        new_genome = deepcopy(best_genomes[0])
        a_genome = np.random.choice(best_genomes)
        b_genome = np.random.choice(best_genomes)

        for j in range(input_length):
            cut = np.random.randint(new_genome.w1_A.shape[1])
            new_genome.w1_A[j, :cut] = a_genome.w1_A[j, :cut]
            new_genome.w1_A[j, cut:] = b_genome.w1_A[j, cut:]

        for j in range(h1):
            cut = np.random.randint(new_genome.w2_A.shape[1])
            new_genome.w2_A[j, :cut] = a_genome.w2_A[j, :cut]
            new_genome.w2_A[j, cut:] = b_genome.w2_A[j, cut:]

        for j in range(h2):
            cut = np.random.randint(new_genome.w3_A.shape[1])
            new_genome.w3_A[j, :cut] = a_genome.w3_A[j, :cut]
            new_genome.w3_A[j, cut:] = b_genome.w3_A[j, cut:]

        for j in range(h3):
            cut = np.random.randint(new_genome.w4_A.shape[1])
            new_genome.w4_A[j, :cut] = a_genome.w4_A[j, :cut]
            new_genome.w4_A[j, cut:] = b_genome.w4_A[j, cut:]

        for j in range(input_length+18):
            cut = np.random.randint(new_genome.w5_A.shape[1])
            new_genome.w5_A[j, :cut] = a_genome.w5_A[j, :cut]
            new_genome.w5_A[j, cut:] = b_genome.w5_A[j, cut:]

        for j in range(h1):
            cut = np.random.randint(new_genome.w6_A.shape[1])
            new_genome.w6_A[j, :cut] = a_genome.w6_A[j, :cut]
            new_genome.w6_A[j, cut:] = b_genome.w6_A[j, cut:]

        for j in range(h2):
            cut = np.random.randint(new_genome.w7_A.shape[1])
            new_genome.w7_A[j, :cut] = a_genome.w7_A[j, :cut]
            new_genome.w7_A[j, cut:] = b_genome.w7_A[j, cut:]

        for j in range(h3):
            cut = np.random.randint(new_genome.w8_A.shape[1])
            new_genome.w8_A[j, :cut] = a_genome.w8_A[j, :cut]
            new_genome.w8_A[j, cut:] = b_genome.w8_A[j, cut:]

        # Line B
        for j in range(input_length+19):
            cut = np.random.randint(new_genome.w1_B.shape[1])
            new_genome.w1_B[j, :cut] = a_genome.w1_B[j, :cut]
            new_genome.w1_B[j, cut:] = b_genome.w1_B[j, cut:]

        for j in range(h1):
            cut = np.random.randint(new_genome.w2_B.shape[1])
            new_genome.w2_B[j, :cut] = a_genome.w2_B[j, :cut]
            new_genome.w2_B[j, cut:] = b_genome.w2_B[j, cut:]

        for j in range(h2):
            cut = np.random.randint(new_genome.w3_B.shape[1])
            new_genome.w3_B[j, :cut] = a_genome.w3_B[j, :cut]
            new_genome.w3_B[j, cut:] = b_genome.w3_B[j, cut:]

        for j in range(h3):
            cut = np.random.randint(new_genome.w4_B.shape[1])
            new_genome.w4_B[j, :cut] = a_genome.w4_B[j, :cut]
            new_genome.w4_B[j, cut:] = b_genome.w4_B[j, cut:]

        for j in range(input_length+37):
            cut = np.random.randint(new_genome.w5_B.shape[1])
            new_genome.w5_B[j, :cut] = a_genome.w5_B[j, :cut]
            new_genome.w5_B[j, cut:] = b_genome.w5_B[j, cut:]

        for j in range(h1):
            cut = np.random.randint(new_genome.w6_B.shape[1])
            new_genome.w6_B[j, :cut] = a_genome.w6_B[j, :cut]
            new_genome.w6_B[j, cut:] = b_genome.w6_B[j, cut:]

        for j in range(h2):
            cut = np.random.randint(new_genome.w7_B.shape[1])
            new_genome.w7_B[j, :cut] = a_genome.w7_B[j, :cut]
            new_genome.w7_B[j, cut:] = b_genome.w7_B[j, cut:]

        for j in range(h3):
            cut = np.random.randint(new_genome.w8_B.shape[1])
            new_genome.w8_B[j, :cut] = a_genome.w8_B[j, :cut]
            new_genome.w8_B[j, cut:] = b_genome.w8_B[j, cut:]

        best_genomes.append(new_genome)

    # 모델 초기화
    genomes = []
    for i in range(int(N_POPULATION / len(best_genomes))):
        for bg in best_genomes:
            new_genome = deepcopy(bg)
            mean = 0
            stddev = 0.2
            # 50% 확률로 모델 변형
            if np.random.uniform(0, 1) < PROB_MUTATION:
                new_genome.w1_A += new_genome.w1_A * np.random.normal(mean, stddev, size=(input_length, h1)) * np.random.randint(0, 2, (input_length, h1))
            if np.random.uniform(0, 1) < PROB_MUTATION:
                new_genome.w1_B += new_genome.w1_B * np.random.normal(mean, stddev, size=(input_length+19, h1)) * np.random.randint(0, 2, (input_length+19, h1))
            if np.random.uniform(0, 1) < PROB_MUTATION:
                new_genome.w2_A += new_genome.w2_A * np.random.normal(mean, stddev, size=(h1, h2)) * np.random.randint(0, 2, (h1, h2))
            if np.random.uniform(0, 1) < PROB_MUTATION:
                new_genome.w2_B += new_genome.w2_B * np.random.normal(mean, stddev, size=(h1, h2)) * np.random.randint(0, 2,(h1, h2))
            if np.random.uniform(0, 1) < PROB_MUTATION:
                new_genome.w3_A += new_genome.w3_A * np.random.normal(mean, stddev, size=(h2, h3)) * np.random.randint(0, 2,(h2, h3))
            if np.random.uniform(0, 1) < PROB_MUTATION:
                new_genome.w3_B += new_genome.w3_B * np.random.normal(mean, stddev, size=(h2, h3)) * np.random.randint(0, 2,(h2,h3))
            if np.random.uniform(0, 1) < PROB_MUTATION:
                new_genome.w4_A += new_genome.w4_A * np.random.normal(mean, stddev, size=(h3, output_length_1)) * np.random.randint(0, 2,(h3, output_length_1))
            if np.random.uniform(0, 1) < PROB_MUTATION:
                new_genome.w4_B += new_genome.w4_B * np.random.normal(mean, stddev, size=(h3, output_length_1)) * np.random.randint(0, 2, (h3, output_length_1))
            if np.random.uniform(0, 1) < PROB_MUTATION:
                new_genome.w5_A += new_genome.w5_A * np.random.normal(mean, stddev, size=(input_length+18, h1)) * np.random.randint(0, 2, (input_length+18, h1))
            if np.random.uniform(0, 1) < PROB_MUTATION:
                new_genome.w5_B += new_genome.w5_B * np.random.normal(mean, stddev, size=(input_length+37, h1)) * np.random.randint(0, 2,(input_length+37, h1))
            if np.random.uniform(0, 1) < PROB_MUTATION:
                new_genome.w6_A += new_genome.w6_A * np.random.normal(mean, stddev, size=(h1, h2)) * np.random.randint(0, 2, (h1, h2))
            if np.random.uniform(0, 1) < PROB_MUTATION:
                new_genome.w6_B += new_genome.w6_B * np.random.normal(mean, stddev, size=(h1, h2)) * np.random.randint(0, 2, (h1, h2))
            if np.random.uniform(0, 1) < PROB_MUTATION:
                new_genome.w7_A += new_genome.w7_A * np.random.normal(mean, stddev, size=(h2, h3)) * np.random.randint(0, 2, (h2, h3))
            if np.random.uniform(0, 1) < PROB_MUTATION:
                new_genome.w7_B += new_genome.w7_B * np.random.normal(mean, stddev, size=(h2, h3)) * np.random.randint(0, 2, (h2, h3))
            if np.random.uniform(0, 1) < PROB_MUTATION:
                new_genome.w8_A += new_genome.w8_A * np.random.normal(mean, stddev, size=(h3, output_length_2)) * np.random.randint(0, 2, (h3, output_length_2))
            if np.random.uniform(0, 1) < PROB_MUTATION:
                new_genome.w8_B += new_genome.w8_B * np.random.normal(mean, stddev, size=(h3, output_length_2)) * np.random.randint(0, 2, (h3, output_length_2))
            genomes.append(new_genome)

    if REVERSE:
        if bs < score_lower_bound_for_new_half:
            genomes[len(genomes) // 2:] = [Genome(score_ini, input_length, output_length_1, output_length_2, h1, h2, h3)
                                           for _ in range(N_POPULATION // 2)]
    else:
        if bs > score_lower_bound_for_new_half:
            genomes[len(genomes) // 2:] = [Genome(score_ini, input_length, output_length_1, output_length_2, h1, h2, h3)
                                           for _ in range(N_POPULATION // 2)]

    n_gen += 1


## 6. 결과 및 결언
## Conclusion & Discussion


### 결과 그래프
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Score Graph
score_history = np.array(score_history)
high_score_history = np.array(high_score_history)
mean_score_history = np.array(mean_score_history)

plt.plot(score_history[:, 0], score_history[:, 1], '-o', label='BEST')
plt.plot(high_score_history[:, 0], high_score_history[:, 1], '-o', label='High')
plt.plot(mean_score_history[:, 0], mean_score_history[:, 1], '-o', label='Mean')
plt.legend()
plt.xlim(0, EPOCHS)
plt.ylim(bottom=0)
plt.xlabel('Epochs')
plt.ylabel('Score')
# plt.show()
plt.savefig("{}v_submission_bestscore_{:.5f}_{}.png".format(__version__, best_genomes[0].score, dt_string))

### Create a submission file
submission_file = '{}v_submission_bestscore_{:.5f}_{}.csv'.format(__version__, best_genomes[0].score, dt_string)
logging.info("Save submission to {}".format(submission_file))
save_submission_file(best_genomes, submission_file)

# Save best genomes
import pickle
for i in range(N_BEST):
    best_genome_file = "{}v_best_genomes_{:.5f}_{}.pkl".format(__version__, best_genomes[0].score, i)
    logging.info("Save best genoms to {}".format(best_genome_file))
    with open(best_genome_file, "wb") as output:
        pickle.dump(best_genomes[i], output, pickle.HIGHEST_PROTOCOL)


# 91.26 점은 23*24까지 MOL을 0으로 채운 결과임. (코드 이후에 변경했음)

# 새롭게 적용한 내용들
# 얼리스탑
# 이전 지놈 저장해놓고 이용하기 (베스트 지놈 저장)
# 로그 기록
# 서치 속도 향상을 위해 기준점수 미달 시 절반은 새로운 (랜덤) 유전자 생성 (원래는 스코어 이닛보다 작을 경우 였음)
# Line A, B 을 따로 학습 (뉴럴넷은 같은 사이즈)
# CHANGE, STOP 적용
# 스코어 목적함수 변경
# 학습 에폭을 많이 했음. (이유는 계속 성장이 보여서, 즉 수렴을 아직 안했기 때문)
# 한계점 1: 시작 재고를 기준으로 했을때 가능한 점수가 100점이 안됨. 남는게 무조건 생김. Upper bound를 계산해볼 수 있다.
# 한계점 2: Change, Stop의 패널티가 커서 높은 점수를 얻은 경우를 보면 change, stop을 사용하지 않게 된다. (이것이 의도된 것인지, 목적함수 설정의 문제인지)
# 4월 15~23일에 PRT 재고가 있는 경우 MOL 생산이 가능하도록 변경
# 현재 상태 재고를 뉴럴넷의 인풋으로 넣어줌 (A와 B가 연결됨)
# 수요가 18시 기준이므로, 뉴럴넷의 인풋도 18시 기준으로 수요를 반영함.
# 수요분을 너무 많이 보는 듯 함. 2~3일치의 수요만 미리 봐도 수요 충족 가능. 30일치는 너무 많은 듯 하여 10일치 정도로 줄여 학습 속도를 향상시킴
# MOL 재고를 현시간의 BLK재고로 환산하여 stock을 채우고, 뉴럴넷의 인풋으로 BLK 재고만 넣어줌

# 개선해야 할점
# 코드 정리, optimization
# 업데이트를 두군데서 해서 문제임. update mask, predict
# 변수명 문제. possible ot process -> not required check
# CHANGE 뉴럴넷을 따로 둘 수 있음. (CHANGE 이벤트 수가 너무 많으므로, CHANGE를 하나로 두고, 그 중에 선택 가능한 CHANGE를 예측하는 뉴럴넷)
# MOL개수를 실수로 예측하는 뉴럴넷으로 변경 가능. 현재는 0.5단위 카테고리 변수.
# overfitting 하는 듯 하다? (러닝할수록 더 높아짐)
# [v] Input을 현재 오더 뿐만 아니라 재고도 고려해서 해야 할듯. 물론 목적함수에서 이게 적용되고 있긴 함.
# 15~23일 일때, change to X에서 X재고가 없는 경우 애초에 event mask를 False로 강제할 수 있음. (이 경우에는 change는 패널티만 증가시킴)