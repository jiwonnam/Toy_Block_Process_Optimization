__author__ = "Jonghun Park"
__date__ = 6 / 5 / 20
__email__ = "jonghunpark@ucsd.edu"

import pandas as pd

# Load best genomes
import pickle
best_genomes = []
N_BEST = 1
PRETRAINED_BEST_SCORE = 56.70759
for i in range(N_BEST):
    with open("best_genomes_{}_{}.pkl".format(PRETRAINED_BEST_SCORE, i), "rb") as input:
        best_genomes.append(pickle.load(input))
        print("Loaded {}th best genome, score: {}".format(i, best_genomes[i].score))


from module.simulator import Simulator
import numpy as np

simulator = Simulator()
order = pd.read_csv('module/order.csv')
submission = best_genomes[0].predict(order)
score, df_stock = simulator.get_score(submission)

# PRTs = df_stock[['PRT_1', 'PRT_2', 'PRT_3', 'PRT_4']].values
# PRTs = (PRTs[:-1] - PRTs[1:])[24 * 23:]
# PRTs = np.ceil(PRTs * 1.1)
# PAD = np.zeros((24 * 23 + 1, 4))
# PRTs = np.append(PRTs, PAD, axis=0).astype(int)
# # Submission 파일에 PRT 입력
# submission.loc[:, 'PRT_1':'PRT_4'] = PRTs
# submission.to_csv("52.01471.csv", index=False)
print(score)
