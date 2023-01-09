# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
try:
    import queue
except ImportError:
    import Queue as queue
from pathlib import Path
from module.simulator import Simulator
from collections import defaultdict


submission_ini = pd.read_csv('module/sample_submission.csv')

class Genome():
    def __init__(self, score_ini, input_len, output_len_1, output_len_2, h1=50, h2=50, h3=50):
        # 평가 점수 초기화
        self.score = score_ini
        
        # 히든레이어 노드 개수
        self.hidden_layer1 = h1
        self.hidden_layer2 = h2
        self.hidden_layer3 = h3
        
        # Event_A 신경망 가중치 생성
        self.w1_A = np.random.randn(input_len, self.hidden_layer1)
        self.w2_A = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w3_A = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w4_A = np.random.randn(self.hidden_layer3, output_len_1)
        
        # MOL_A 수량 신경망 가중치 생성 TODO: Could be a regression model maximum 5.7
        self.w5_A = np.random.randn(input_len+18, self.hidden_layer1) # add the event to input
        self.w6_A = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w7_A = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w8_A = np.random.randn(self.hidden_layer3, output_len_2)

        # Event_B 신경망 가중치 생성
        self.w1_B = np.random.randn(input_len+19, self.hidden_layer1)
        self.w2_B = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w3_B = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w4_B = np.random.randn(self.hidden_layer3, output_len_1)

        # MOL_B 수량 신경망 가중치 생성 TODO: Could be a regression model maximum 5.7
        self.w5_B = np.random.randn(input_len+37, self.hidden_layer1) # add the event to input
        self.w6_B = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w7_B = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w8_B = np.random.randn(self.hidden_layer3, output_len_2)
        
        # Event_A 종류
        event_size = output_len_1
        self.event_mask_A = np.zeros([event_size], np.bool)  # 가능한 이벤트 검사용 마스크

        self.possible_events = []
        self.possible_events.append('STOP')
        for i in range(1, 5):
            self.possible_events.append('CHECK_{}'.format(i))
        for i in range(1, 5):
            for j in range(1, 5):
                if i != j:
                    self.possible_events.append('CHANGE_{}{}'.format(i, j))
        self.possible_events.append('PROCESS')
        self.index_to_event = {i: self.possible_events[i] for i in range(len(self.possible_events))}

        self.event_to_index = {}
        for index, event in self.index_to_event.items():
            self.event_to_index[event] = index

        self.change_time = defaultdict(dict)
        self.change_time[1][2] = 6
        self.change_time[1][3] = 13
        self.change_time[1][4] = 13
        self.change_time[2][1] = 6
        self.change_time[2][3] = 13
        self.change_time[2][4] = 13
        self.change_time[3][1] = 13
        self.change_time[3][2] = 13
        self.change_time[3][4] = 6
        self.change_time[4][1] = 13
        self.change_time[4][2] = 13
        self.change_time[4][3] = 6

        self.check_time_A = 28    # 28시간 검사를 완료했는지 검사, CHECK Event시 -1, processtime_time >=98 이면 28
        self.possible_to_process_A = 0        # 생산 가능 여부, 0 이면 28 시간 검사 필요
        self.process_product_number_A = 0   # 생산 물품 번호 1-4,
        self.process_time_A = 0   # 생산시간이 얼마나 지속되었는지 검사, PROCESS +1, CHANGE +1, 최대 140

        self.is_changing_A = False  # 현재 변경중인지를 나타내는 변수
        self.to_change_A = 0  # 어떤 생산 물품 번호로 바꾸는 중인지
        self.change_time_A = 0  # 변경시간이 얼마나 지속되었는지 검사, CHANGE +1, change_time까지 지속
        self.previous_event_A = ""

        self.is_stopping_A = False  # 현재 스탑중인지를 나타내는 변수
        self.stop_time_A = 0  # 스탑 시간

        # Event_B 종류
        self.event_mask_B = np.zeros([event_size], np.bool)  # 가능한 이벤트 검사용 마스크

        self.check_time_B = 28  # 28시간 검사를 완료했는지 검사, CHECK Event시 -1, processtime_time >=98 이면 28
        self.possible_to_process_B = 0  # 생산 가능 여부, 0 이면 28 시간 검사 필요
        self.process_product_number_B = 0  # 생산 물품 번호 1-4,
        self.process_time_B = 0  # 생산시간이 얼마나 지속되었는지 검사, PROCESS +1, CHANGE +1, 최대 140

        self.is_changing_B = False  # 현재 변경중인지를 나타내는 변수
        self.to_change_B = 0  # 어떤 생산 물품 번호로 바꾸는 중인지
        self.change_time_B = 0  # 변경시간이 얼마나 지속되었는지 검사, CHANGE +1, change_time까지 지속
        self.previous_event_B = ""

        self.is_stopping_B = False  # 현재 스탑중인지를 나타내는 변수
        self.stop_time_B = 0  # 스탑 시간

        # Make submission file
        self.submission = submission_ini
        self.submission.loc[:, 'PRT_1':'PRT_4'] = 0

        # Make stock file
        self.df_stock = self.submission.copy()
        column = 'time'
        self.df_stock.index = pd.to_datetime(self.df_stock[column])
        self.df_stock = self.df_stock.drop(self.df_stock.columns, axis=1)
        self.df_stock.index.name = column
        self.df_stock_index_size = len(self.df_stock.index)

        self.daily_order = pd.read_csv('module/order.csv')
        # Add 30days order init after June
        for i in range(30):
            self.daily_order.loc[91 + i, :] = ['0000-00-00', 0, 0, 0, 0]
        self.order = pd.read_csv('module/order.csv', index_col=0)
        self.order.index = pd.to_datetime(self.order.index)
        self.stock_ini = pd.read_csv('module/stock.csv')

    def change_order_file(self, order_file):
        self.daily_order = pd.read_csv(order_file)
        # Add 30days order init after June
        for i in range(30):
            self.daily_order.loc[91 + i, :] = ['0000-00-00', 0, 0, 0, 0]
        self.order = pd.read_csv(order_file, index_col=0)
        self.order.index = pd.to_datetime(self.order.index)

    def update_mask(self):
        # Update line A
        self.event_mask_A[:] = False
        if self.possible_to_process_A == 0:
            if self.check_time_A == 28:
                self.event_mask_A[1:5] = True
            if self.check_time_A < 28:
                self.event_mask_A[self.process_product_number_A] = True
        if self.possible_to_process_A == 1:
            if self.is_changing_A:  # If we are in change, only that change should be True
                self.event_mask_A[self.event_to_index['CHANGE_{}{}'.format(self.process_product_number_A, self.to_change_A)]] = True
            elif self.is_stopping_A:  # In stop process
                if self.stop_time_A < 29:  # Didn't meet minimum 28h
                    self.event_mask_A[self.event_to_index['STOP']] = True
                else:
                    self.event_mask_A[self.event_to_index['STOP']] = True
                    self.event_mask_A[1:5] = True  # CHECKS are possible
            else:  # Process, Check, Change, STOP are all possible.
                self.event_mask_A[self.event_to_index['PROCESS']] = True
                if self.process_time_A != 0 and not self.previous_event_A.startswith("CHANGE"):  # No consecutive events
                    for j in range(1, 5):
                        if self.process_product_number_A == j:
                            continue
                        # if we are in process, but we can go to change, any change is possible
                        if self.change_time[self.process_product_number_A][j] + self.process_time_A <= 140:
                            self.event_mask_A[self.event_to_index['CHANGE_{}{}'.format(self.process_product_number_A, j)]] = True
                if self.process_time_A > 98:
                    self.event_mask_A[self.event_to_index['STOP']] = True  # STOP is also possible
                    self.event_mask_A[1:5] = True  # CHECKS are possible

        # Update line B
        self.event_mask_B[:] = False
        if self.possible_to_process_B == 0:
            if self.check_time_B == 28:
                self.event_mask_B[1:5] = True
            if self.check_time_B < 28:
                self.event_mask_B[self.process_product_number_B] = True
        if self.possible_to_process_B == 1:
            if self.is_changing_B:  # If we are in change, only that change should be True
                self.event_mask_B[self.event_to_index['CHANGE_{}{}'.format(self.process_product_number_B, self.to_change_B)]] = True
            elif self.is_stopping_B:  # In stop process
                if self.stop_time_B < 29:  # Didn't meet minimum 28h
                    self.event_mask_B[self.event_to_index['STOP']] = True
                else:
                    self.event_mask_B[self.event_to_index['STOP']] = True
                    self.event_mask_B[1:5] = True  # CHECKS are possible
            else:  # Process, Check, Change, STOP are all possible.
                self.event_mask_B[self.event_to_index['PROCESS']] = True
                if self.process_time_B != 0 and not self.previous_event_B.startswith("CHANGE"):  # No consecutive events
                    for j in range(1, 5):
                        if self.process_product_number_B == j:
                            continue
                        # if we are in process, but we can go to change, any change is possible
                        if self.change_time[self.process_product_number_B][j] + self.process_time_B <= 140:
                            self.event_mask_B[self.event_to_index['CHANGE_{}{}'.format(self.process_product_number_B, j)]] = True
                if self.process_time_B > 98:
                    self.event_mask_B[self.event_to_index['STOP']] = True  # STOP is also possible
                    self.event_mask_B[1:5] = True  # CHECKS are possible

    
    def forward(self, inputs_event_A):
        # Event_A 신경망
        net = np.matmul(inputs_event_A, self.w1_A)
        net = self.linear(net)
        net = np.matmul(net, self.w2_A)
        net = self.linear(net)
        net = np.matmul(net, self.w3_A)
        net = self.sigmoid(net)
        net = np.matmul(net, self.w4_A)
        net = self.softmax(net)
        net += 1
        net = net * self.event_mask_A
        out1_A = self.index_to_event[np.argmax(net)]

        # Optimization
        # If out1_A is not PROCESS, we don't have to calculate the out2_A because it will be 0
        if out1_A != 'PROCESS':
            out2_A = 0
            one_hot_vector_A = [0] * len(self.possible_events)
            one_hot_vector_A[self.event_to_index[out1_A]] = 1
            inputs_mol_A = np.append(inputs_event_A, one_hot_vector_A) # add event into inputs despite of event kind
        else:
            # MOL_A 수량 신경망
            one_hot_vector_A = [0] * len(self.possible_events)
            one_hot_vector_A[self.event_to_index[out1_A]] = 1
            inputs_mol_A = np.append(inputs_event_A, one_hot_vector_A)
            net = np.matmul(inputs_mol_A, self.w5_A)
            net = self.linear(net)
            net = np.matmul(net, self.w6_A)
            net = self.linear(net)
            net = np.matmul(net, self.w7_A)
            net = self.sigmoid(net)
            net = np.matmul(net, self.w8_A)
            net = self.softmax(net)
            out2_A = np.argmax(net)
            out2_A /= 2

        # Event_B 신경망
        inputs_event_B = np.append(inputs_mol_A, out2_A)
        net = np.matmul(inputs_event_B, self.w1_B)
        net = self.linear(net)
        net = np.matmul(net, self.w2_B)
        net = self.linear(net)
        net = np.matmul(net, self.w3_B)
        net = self.sigmoid(net)
        net = np.matmul(net, self.w4_B)
        net = self.softmax(net)
        net += 1
        net = net * self.event_mask_B
        out1_B = self.index_to_event[np.argmax(net)]

        if out1_B != 'PROCESS':
            out2_B = 0
        else:
            # MOL_B 수량 신경망
            one_hot_vector_B = [0] * len(self.possible_events)
            one_hot_vector_B[self.event_to_index[out1_A]] = 1
            inputs_mol_B = np.append(inputs_event_B, one_hot_vector_B)
            net = np.matmul(inputs_mol_B, self.w5_B)
            net = self.linear(net)
            net = np.matmul(net, self.w6_B)
            net = self.linear(net)
            net = np.matmul(net, self.w7_B)
            net = self.sigmoid(net)
            net = np.matmul(net, self.w8_B)
            net = self.softmax(net)
            out2_B = np.argmax(net)
            out2_B /= 2

        return out1_A, out2_A, out1_B, out2_B

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def linear(self, x):
        return x
    

    def predict(self):
        # Init
        total_orders = 0
        sum_of_stop_time = 0
        stop_count = 0
        sum_of_change_time = 0
        change_count = 0

        # Set the initial stock
        self.df_stock['PRT_1'] = 0.0
        self.df_stock['PRT_2'] = 0.0
        self.df_stock['PRT_3'] = 0.0
        self.df_stock['PRT_4'] = 0.0
        self.df_stock['MOL_1'] = 0.0
        self.df_stock['MOL_2'] = 0.0
        self.df_stock['MOL_3'] = 0.0
        self.df_stock['MOL_4'] = 0.0
        self.df_stock['BLK_1'] = 0.0
        self.df_stock['BLK_2'] = 0.0
        self.df_stock['BLK_3'] = 0.0
        self.df_stock['BLK_4'] = 0.0

        for column in self.df_stock.columns:
            self.df_stock.iloc[0, self.df_stock.columns.get_loc(column)] = self.df_stock.iloc[0, self.df_stock.columns.get_loc(column)] + self.stock_ini.iloc[0][column]

        # Make order and Rescale
        hourly_order = self.df_stock.drop(self.df_stock.columns, axis=1)
        dt_order = pd.Timedelta(hours=18)
        for column in ['BLK_1', 'BLK_2', 'BLK_3', 'BLK_4']:
            for time in self.order.index:
                hourly_order.loc[time + dt_order, column] = self.order.loc[time, column]
        hourly_order = hourly_order.fillna(0)

        # Cut yield and probability
        mol2blk = {}
        mol2blk['MOL_1'] = 'BLK_1'
        mol2blk['MOL_2'] = 'BLK_2'
        mol2blk['MOL_3'] = 'BLK_3'
        mol2blk['MOL_4'] = 'BLK_4'

        cut = {}
        cut['BLK_1'] = 506
        cut['BLK_2'] = 506
        cut['BLK_3'] = 400
        cut['BLK_4'] = 400

        p_BLK = {}
        p_BLK['BLK_1'] = 0.851
        p_BLK['BLK_2'] = 0.901
        blk_diffs = []

        # Initiate process_queue
        process_queue_A = queue.Queue()
        #process_queue_A.maxsize = 48
        for i in range(48):
            process_queue_A.put(('default', 0))

        process_queue_B = queue.Queue()
        #process_queue_B.maxsize = 48
        for i in range(48):
            process_queue_B.put(('default', 0))

        # Fill in submission file
        for s, time in enumerate(self.df_stock.index):

            month = time.month
            if month == 4:
                p_BLK['BLK_3'] = 0.710
                p_BLK['BLK_4'] = 0.700
            elif month == 5:
                p_BLK['BLK_3'] = 0.742
                p_BLK['BLK_4'] = 0.732
            elif month == 6:
                p_BLK['BLK_3'] = 0.759
                p_BLK['BLK_4'] = 0.749
            else:
                p_BLK['BLK_3'] = 0.0
                p_BLK['BLK_4'] = 0.0

            self.update_mask()
            # Order rescale: daily order occurs at 6 pm.
            # Orders from today to 10 days are the inputs
            inputs_event_A = np.array(self.daily_order.loc[(s+5)//24:((s+5)//24+9), 'BLK_1':'BLK_4']).reshape(-1)  #1 x 40
            if s == 0:
                inputs_event_A = np.append(inputs_event_A, self.df_stock.iloc[s, 8:])  # Add BLK stocks
            else:
                inputs_event_A = np.append(inputs_event_A, self.df_stock.iloc[s-1, 8:])  # Add BLK stocks

            out_event_A, out_mol_A, out_event_B, out_mol_B = self.forward(inputs_event_A)

            # Event_A
            if 'CHECK' in out_event_A:
                if self.possible_to_process_A == 1:  # First time when PROCESS, CHANGE, STOP -> Check
                    self.possible_to_process_A = 0
                    self.check_time_A = 28
                    self.is_stopping_A = False
                    self.stop_time_A = 0
                self.check_time_A -= 1
                self.process_product_number_A = int(out_event_A[-1])
                if self.check_time_A == 0:
                    self.possible_to_process_A = 1
                    self.process_time_A = 0
            elif out_event_A == 'PROCESS':
                self.process_time_A += 1
                if self.process_time_A == 140:
                    self.possible_to_process_A = 0
                    self.check_time_A = 28
            elif 'CHANGE' in out_event_A:
                self.to_change_A = int(out_event_A[-1])  # Index of the block (1 to 4)
                self.is_changing_A = True
                self.process_time_A += 1
                self.change_time_A += 1
                if self.change_time_A == self.change_time[int(out_event_A[-2])][int(out_event_A[-1])]:  # if change is done, process is also done
                    self.process_product_number_A = self.to_change_A  # Now our input is product Y, CHANGE_X_Y
                    if self.process_time_A == 140:
                        self.possible_to_process_A = 0
                        self.check_time_A = 28
                    self.change_time_A = 0
                    self.is_changing_A = False
                sum_of_change_time += 1
                if 'CHANGE' not in self.previous_event_A:
                    change_count += 1
            elif 'STOP' == out_event_A:
                self.is_stopping_A = True
                self.stop_time_A += 1
                if self.stop_time_A == 24*8:  # Only CHECK is possible
                    self.possible_to_process_A = 0
                    self.check_time_A = 28
                    self.is_stopping_A = False
                    self.stop_time_A = 0
                sum_of_stop_time += 1
                if 'STOP' != self.previous_event_A:
                    stop_count += 1
            else:
                print("ERROR: Event Name: {}".format(out_event_A))
                exit(1)

            # Fill A Event and Mol, and Update output amount
            self.submission.loc[s, 'Event_A'] = out_event_A
            self.previous_event_A = out_event_A

            p_MOL = 0.975
            if out_event_A == 'PROCESS':
                queue_result_A = process_queue_A.get()
                queue_which_mol_A = queue_result_A[0]
                queue_mol_count_A = queue_result_A[1]
                if s < 24*14:  # TODO consider daily MOL max count
                    self.submission.loc[s, 'MOL_A'] = 0
                    process_queue_A.put(('Impossible', 0))
                elif 24*14 <= s < 24*23:  # From 0 day to 23 days, we can put MOL_X only if there is stock for the PRT_X # TODO consider daily MOL max count
                    if self.df_stock.loc[time - pd.Timedelta(hours=1), 'PRT_' + str(self.process_product_number_A)] - out_mol_A >= 0:
                        self.submission.loc[s, 'MOL_A'] = out_mol_A
                        self.df_stock.loc[time, 'PRT_' + str(self.process_product_number_A)] += (-1) * out_mol_A
                        process_queue_A.put(('MOL_' + str(self.process_product_number_A), out_mol_A))
                        if queue_mol_count_A > 0:
                            produced_mol_A = queue_mol_count_A * p_MOL
                            produced_blk_col = mol2blk[queue_which_mol_A]
                            produced_blk_A = produced_mol_A * cut[produced_blk_col] * p_BLK[produced_blk_col]
                            self.df_stock.loc[time, produced_blk_col] += produced_blk_A
                            self.df_stock.loc[time, queue_which_mol_A] = 0  # All MOL has cut to BLK, so set it as 0
                    else:
                        if self.df_stock.loc[time - pd.Timedelta(hours=1), 'PRT_' + str(self.process_product_number_A)] > 0:
                            self.submission.loc[s, 'MOL_A'] = self.df_stock.loc[time - pd.Timedelta(hours=1), 'PRT_' + str(self.process_product_number_A)]
                            self.df_stock.loc[time, 'PRT_' + str(self.process_product_number_A)] += (-1) * self.df_stock.loc[time - pd.Timedelta(hours=1), 'PRT_' + str(self.process_product_number_A)]
                            process_queue_A.put(('MOL_' + str(self.process_product_number_A), self.df_stock.loc[time - pd.Timedelta(hours=1), 'PRT_' + str(self.process_product_number_A)]))
                            if queue_mol_count_A > 0:
                                produced_mol_A = queue_mol_count_A * p_MOL
                                produced_blk_col = mol2blk[queue_which_mol_A]
                                produced_blk_A = produced_mol_A * cut[produced_blk_col] * p_BLK[produced_blk_col]
                                self.df_stock.loc[time, produced_blk_col] += produced_blk_A
                                self.df_stock.loc[time, queue_which_mol_A] = 0  # All MOL has cut to BLK, so set it as 0
                        else:
                            self.submission.loc[s, 'MOL_A'] = 0
                            process_queue_A.put(('MOL_' + str(self.process_product_number_A), 0))
                            if queue_mol_count_A > 0:
                                produced_mol_A = queue_mol_count_A * p_MOL
                                produced_blk_col = mol2blk[queue_which_mol_A]
                                produced_blk_A = produced_mol_A * cut[produced_blk_col] * p_BLK[produced_blk_col]
                                self.df_stock.loc[time, produced_blk_col] += produced_blk_A
                                self.df_stock.loc[time, queue_which_mol_A] = 0  # All MOL has cut to BLK, so set it as 0
                else:
                    self.submission.loc[s, 'MOL_A'] = out_mol_A
                    self.df_stock.loc[time, 'PRT_' + str(self.process_product_number_A)] += (-1) * out_mol_A
                    process_queue_A.put(('MOL_' + str(self.process_product_number_A), out_mol_A))
                    if queue_mol_count_A > 0:
                        produced_mol_A = queue_mol_count_A * p_MOL
                        produced_blk_col = mol2blk[queue_which_mol_A]
                        produced_blk_A = produced_mol_A * cut[produced_blk_col] * p_BLK[produced_blk_col]
                        self.df_stock.loc[time, produced_blk_col] += produced_blk_A
                        self.df_stock.loc[time, queue_which_mol_A] = 0  # All MOL has cut to BLK, so set it as 0
            else:
                self.submission.loc[s, 'MOL_A'] = 0

            # Event_B
            if 'CHECK' in out_event_B:
                if self.possible_to_process_B == 1:  # First time when PROCESS, CHANGE, STOP -> Check
                    self.possible_to_process_B = 0
                    self.check_time_B = 28
                    self.is_stopping_B = False
                    self.stop_time_B = 0
                self.check_time_B -= 1
                self.process_product_number_B = int(out_event_B[-1])
                if self.check_time_B == 0:
                    self.possible_to_process_B = 1
                    self.process_time_B = 0
            elif out_event_B == 'PROCESS':
                self.process_time_B += 1
                if self.process_time_B == 140:
                    self.possible_to_process_B = 0
                    self.check_time_B = 28
            elif 'CHANGE' in out_event_B:
                self.to_change_B = int(out_event_B[-1])  # Index of the block (1 to 4)
                self.is_changing_B = True
                self.process_time_B += 1
                self.change_time_B += 1
                if self.change_time_B == self.change_time[int(out_event_B[-2])][int(out_event_B[-1])]: # if change is done, process is also done
                    self.process_product_number_B = self.to_change_B  # Now our input is product Y, CHANGE_X_Y
                    if self.process_time_B == 140:
                        self.possible_to_process_B = 0
                        self.check_time_B = 28
                    self.change_time_B = 0
                    self.is_changing_B = False
                sum_of_change_time += 1
                if 'CHANGE' not in self.previous_event_B:
                    change_count += 1
            elif 'STOP' == out_event_B:
                self.is_stopping_B = True
                self.stop_time_B += 1
                if self.stop_time_B == 24*8:  # Only CHECK is possible
                    self.possible_to_process_B = 0
                    self.check_time_B = 28
                    self.is_stopping_B = False
                    self.stop_time_B = 0
                sum_of_stop_time += 1
                if 'STOP' != self.previous_event_B:
                    stop_count += 1
            else:
                print("ERROR: Event Name: {}".format(out_event_B))
                exit(1)

            # Fill B Event and Mol, and Update output amount
            self.submission.loc[s, 'Event_B'] = out_event_B
            self.previous_event_B = out_event_B

            if out_event_B == 'PROCESS':
                queue_result_B = process_queue_B.get()
                queue_which_mol_B = queue_result_B[0]
                queue_mol_count_B = queue_result_B[1]
                if s < 24*14:  # TODO consider daily MOL max count
                    self.submission.loc[s, 'MOL_B'] = 0
                    process_queue_B.put(('Impossible', 0))
                elif 24*14 <= s < 24*23:  # TODO consider daily MOL max count
                    if self.df_stock.loc[time - pd.Timedelta(hours=1), 'PRT_' + str(self.process_product_number_B)] - out_mol_B >= 0:
                        self.submission.loc[s, 'MOL_B'] = out_mol_B
                        self.df_stock.loc[time, 'PRT_' + str(self.process_product_number_B)] += (-1) * out_mol_B
                        process_queue_B.put(('MOL_' + str(self.process_product_number_B), out_mol_B))
                        if queue_mol_count_B > 0:
                            produced_mol_B = queue_mol_count_B * p_MOL
                            produced_blk_col = mol2blk[queue_which_mol_B]
                            produced_blk_B = produced_mol_B * cut[produced_blk_col] * p_BLK[produced_blk_col]
                            self.df_stock.loc[time, produced_blk_col] += produced_blk_B
                            self.df_stock.loc[time, queue_which_mol_B] = 0  # All MOL has cut to BLK, so set it as 0
                    else:
                        if self.df_stock.loc[time - pd.Timedelta(hours=1), 'PRT_' + str(self.process_product_number_B)] > 0:
                            self.submission.loc[s, 'MOL_B'] = self.df_stock.loc[time - pd.Timedelta(hours=1), 'PRT_' + str(self.process_product_number_B)]
                            self.df_stock.loc[time, 'PRT_' + str(self.process_product_number_B)] += (-1) * self.df_stock.loc[time - pd.Timedelta(hours=1), 'PRT_' + str(self.process_product_number_B)]
                            process_queue_B.put(('MOL_' + str(self.process_product_number_B), self.df_stock.loc[time - pd.Timedelta(hours=1), 'PRT_' + str(self.process_product_number_B)]))
                            if queue_mol_count_B > 0:
                                produced_mol_B = queue_mol_count_B * p_MOL
                                produced_blk_col = mol2blk[queue_which_mol_B]
                                produced_blk_B = produced_mol_B * cut[produced_blk_col] * p_BLK[produced_blk_col]
                                self.df_stock.loc[time, produced_blk_col] += produced_blk_B
                                self.df_stock.loc[time, queue_which_mol_B] = 0  # All MOL has cut to BLK, so set it as 0
                        else:
                            self.submission.loc[s, 'MOL_B'] = 0
                            process_queue_B.put(('MOL_' + str(self.process_product_number_B), 0))
                            if queue_mol_count_B > 0:
                                produced_mol_B = queue_mol_count_B * p_MOL
                                produced_blk_col = mol2blk[queue_which_mol_B]
                                produced_blk_B = produced_mol_B * cut[produced_blk_col] * p_BLK[produced_blk_col]
                                self.df_stock.loc[time, produced_blk_col] += produced_blk_B
                                self.df_stock.loc[time, queue_which_mol_B] = 0  # All MOL has cut to BLK, so set it as 0
                else:
                    self.submission.loc[s, 'MOL_B'] = out_mol_B
                    self.df_stock.loc[time, 'PRT_' + str(self.process_product_number_B)] += (-1) * out_mol_B
                    process_queue_B.put(('MOL_' + str(self.process_product_number_B), out_mol_B))
                    if queue_mol_count_B > 0:
                        produced_mol_B = queue_mol_count_B * p_MOL
                        produced_blk_col = mol2blk[queue_which_mol_B]
                        produced_blk_B = produced_mol_B * cut[produced_blk_col] * p_BLK[produced_blk_col]
                        self.df_stock.loc[time, produced_blk_col] += produced_blk_B
                        self.df_stock.loc[time, queue_which_mol_B] = 0  # All MOL has cut to BLK, so set it as 0
            else:
                self.submission.loc[s, 'MOL_B'] = 0

            # Reflect the previous stock and Update stock
            if s != 0:
                self.df_stock.loc[time] = self.df_stock.loc[time - pd.Timedelta(hours=1)] + self.df_stock.loc[time]

            # when order exists, cut mol and calculate blk diffs
            for column in hourly_order.columns:
                val = hourly_order.loc[time, column]
                total_orders += val
                if val > 0:
                    blk_stock = self.df_stock.loc[time, column]
                    blk_diff = blk_stock - val

                    self.df_stock.loc[time, column] = blk_diff
                    blk_diffs.append(blk_diff)

        # Delete cells after 6.30
        self.df_stock = self.df_stock.drop(self.df_stock.index[self.df_stock_index_size:])

        # 변수 초기화
        self.check_time_A = 28
        self.possible_to_process_A = 0
        self.process_product_number_A = 0
        self.process_time_A = 0

        self.is_changing_A = False  # 현재 변경중인지를 나타내는 변수
        self.to_change_A = 0  # 어떤 생산 물품 번호로 바꾸는 중인지
        self.change_time_A = 0  # 변경시간이 얼마나 지속되었는지 검사, CHANGE +1, change_time까지 지속

        self.is_stopping_A = False
        self.stop_time_A = 0

        self.check_time_B = 28
        self.possible_to_process_B = 0
        self.process_product_number_B = 0
        self.process_time_B = 0

        self.is_changing_B = False  # 현재 변경중인지를 나타내는 변수
        self.to_change_B = 0  # 어떤 생산 물품 번호로 바꾸는 중인지
        self.change_time_B = 0  # 변경시간이 얼마나 지속되었는지 검사, CHANGE +1, change_time까지 지속

        self.is_stopping_B = False
        self.stop_time_B = 0

        return self.submission, self.df_stock, total_orders, blk_diffs, sum_of_change_time, sum_of_stop_time, change_count, stop_count

def genome_score(genome):
    submission, df_stock, total_orders, blk_diffs, sum_of_change_time, sum_of_stop_time, change_count, stop_count = genome.predict()
    genome.submission = submission
    genome.df_stock = df_stock
    total_time = genome.df_stock_index_size
    genome.score = get_score(total_orders, blk_diffs, total_time, sum_of_change_time, sum_of_stop_time, change_count, stop_count)

    return genome

def get_score(total_orders, blk_diffs, total_time, sum_of_change_time, sum_of_stop_time, change_count, stop_count):
    # F(x, a): 1 - x/a if x < a else 0
    # p: 수요 발생 시 블럭 장난감 생산 부족분 합계
    # q: 수요 발생 시 블럭 장난감 생산 초과분 합계
    # c: 성형 공정 변경 시간 합계
    # c_n: 성형 공정 변경 이벤트 횟수
    # s: 멈춤 시간 합계
    # s_n: 멈춤 이벤트 횟수
    # N: 블럭 장난감 총 수요
    # M: 전체 시간 (Total time * 2?)
    # Score = 50 x F(p, 10N) + 20 x F(q, 10N) + 20 x F(c, M) / (1+0.1 x c_n) + 10 x F(s, M) / (1+0.1 x s_n)

    # Block Order Difference
    blk_shortage = 0.0
    blk_surplus = 0.0
    for item in blk_diffs:
        if item < 0:
            blk_shortage += -item
        else:
            blk_surplus += item

    score = 0
    if blk_shortage < 10 * total_orders:
        score += 50 * (1.0 - blk_shortage / (10 * total_orders))
    if blk_surplus < 10 * total_orders:
        score += 20 * (1.0 - blk_surplus / (10 * total_orders))
    if sum_of_change_time < 2 * total_time:
        score += 20 * (1.0 - sum_of_change_time / (2 * total_time)) / (1.0 + 0.1 * change_count)
    if sum_of_stop_time < 2 * total_time:
        score += 10 * (1.0 - sum_of_stop_time / (2 * total_time)) / (1.0 + 0.1 * stop_count)

    return score
