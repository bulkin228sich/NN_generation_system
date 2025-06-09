import numpy as np

import os
import torch
import torch.nn as nn
import pandas as pd
from models import *


class TradeGame():
    def __init__(self, model, data_file, sort_=30, step=0.01, window_size=180, step_size=[1, 5, 10], cash=1000):
        super().__init__()
        self.model = model
        self.window_size = window_size
        self.step_size = step_size
        self.coin = 0
        self.data_file = data_file
        self.num_str = 0
        self.data = []
        self.cost = 83409.2
        self.norm = []
        self.future_cost = 0
        self.number_order = 0
        self.sort = [[cash, self.coin, i] for i in range(sort_)]
        self.step = step

    def game(self, time):
        # val = [float('inf'), float('-inf')]  # [min, max]
        # for i in range(0, 1000):
        #     self.data = self.data_file[0 + i:180 + i]
        #     self.data = self.preprocess_window(self.data)
        #     self.data = torch.tensor(self.data, dtype=torch.float32).unsqueeze(0)
        #     self.cost = self.data_file[180 + i][-3]
        #     self.future_cost = self.data_file[181 + i][-3]
        #     with torch.no_grad():
        #         out = model(self.data).tolist()[0]  # [p1, p2, p3]
        #         val[0] = min(val[0], out[0])
        #         val[1] = max(val[1], out[0])
        #
        # min_val, max_val = val
        # range_val = max_val - min_val if max_val != min_val else 1e-6

        # result = []
        # for i in range(0, 20):
        #     sdvig = 0
        #     self.data = self.data_file[0 + i + sdvig:180 + i + sdvig]
        #     self.data = self.preprocess_window(self.data)
        #     self.data = torch.tensor(self.data, dtype=torch.float32).unsqueeze(0)
        #     self.cost = self.data_file[180 + i + sdvig][-3]
        #     self.future_cost = self.data_file[181 + i + sdvig][-3]
        #     with torch.no_grad():
        #         self.data = self.data.to(torch.device("cuda"))
        #         out = self.model(self.data).tolist()[0]  # (1,3)
        #         y_model = out[0]
        #     if i == 20:
        #         self.norm = set(self.norm)
        #         if len(self.norm) < 10:
        #             print("no answer")
        #             return ["no answer", *result]
        #     elif i < 20:
        #         self.norm.append(round(y_model, 1))
        #         result.append([y_model, self.cost, self.future_cost])
        result = []

        for i in range(0, time):
            sdvig = 0
            self.data = self.data_file[0 + i + sdvig:180 + i + sdvig]
            self.data = self.preprocess_window(self.data)
            self.data = torch.tensor(self.data, dtype=torch.float32).unsqueeze(0)
            self.cost = self.data_file[180 + i + sdvig][-3]
            self.future_cost = self.data_file[181 + i + sdvig][-3]
            with torch.no_grad():

                self.data = self.data.to(torch.device("cuda"))
                out = self.model(self.data).tolist()[0]  # (1,3)
                y_model = out[0]
            for j in range(0, len(self.sort)):
                k = 0 + (j * self.step)
                if y_model > k:
                    if self.sort[j][0] > 0:  # self.sort[j][0] -> это cash
                        self.buy(y_model, j)
                else:
                    if self.sort[j][1] > 0:  # self.sort[j][1] -> это coin
                        self.sell(j)

            # print(
            #     f"общая стоимость {self.cash + (self.coin * self.cost)} предсказание {y_model}  цена{self.cost} будет {self.data_file[181 + i + sdvig][-3]} валюта{self.cash} монет {self.coin}")
            # print()
            # print(i, "min")
            # print(self.cash + (self.coin * self.cost))

        # print(
        #     f"общая стоимость {self.cash + (self.coin * self.cost)} предсказание {y_model}  цена{self.cost} будет {self.future_cost} валюта{self.cash} монет {self.coin} количество сделок {self.number_order}" )
        # print(j)

        self.number_order = 0
        self.coin = 0
        for capital in self.sort:
                capital[0] = round(capital[0]+capital[1] * self.future_cost,1)
                capital[1] = 0
        return [[float(x) for x in inner] for inner in self.sort]

    def buy(self, out, j):
        frac = 0.7 if out > 0.5 else 0.4
        amount_to_spend = self.sort[j][0] * frac
        self.sort[j][1] += amount_to_spend / self.cost
        self.sort[j][0] -= amount_to_spend
        self.number_order += 1

    def sell(self, j):
        self.sort[j][0] += self.sort[j][1] * self.cost
        self.sort[j][1] = 0.0

    def preprocess_window(self, w):
        o = w[:, 0]
        return np.stack([
            (w[:, 1] - o) / o * 100,
            (w[:, 2] - o) / o * 100,
            (w[:, 3] - o) / o * 100,
            w[:, 4],
            w[:, 5] / 1e6
        ], axis=1)
