import json
import os
import time
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from tqdm import tqdm

from trade_game import TradeGame
from models import *
from train import train_model
from schedule import models_future

start_time = datetime.now()
start_ts = time.time()
result = []
train_losses_models = []
data_file = "btc_10180.csv"
x = np.loadtxt(os.path.join("sup_code", data_file), delimiter=',')
MODEL_BASE = r'saved_models'

seq_len = 180
input_size = 5
out_d = 1
num_layers=6
hidden_size =[512]
dropout = 0.1
for i in hidden_size:
    for j in range(1, num_layers):
        test_data = [[0, p] for p in range(30)]
        for l in range(10):
            param = [seq_len, input_size,  out_d, j, i, dropout]
            m_list = [
                      #(LinearModel(*param), "LinearModel"),
                      #(MLPModel(*param), "MLPModel"),
                      [GRUModel(*param), "GRUModel"],  # 2
                      # [EnhancedRNNModel(*param), "EnhancedRNNModel"],  # 3
                      # [TransformerModel(*param), "TransformerModel"],  # 4
                      # [EnhancedRNNModelV2(*param), "EnhancedRNNModelV2"],  # 5
                      # [TCNModel(*param), "TCNModel"],  # 6
                      # [NBeatsModel(*param), "NBeatsModel"],  # 7
                      # [SimpleInformer(*param), "SimpleInformer"],  # 8
                      # [TFTLite(*param), "TFTLite"],  # 9
                      # [SimpleDeepAR(*param), "SimpleDeepAR"],  # 10
                      # [PatchTST(*param), "PatchTST"],  # 11
                      # [TimesNetClassifier(*param), "TimesNetClassifier"],  # 12
                      # [GRUD(*param), "GRUD"],  # 13
                      # [SCINet(*param), "SCINet"],  # 14
                      # [FEDformer(*param), "FEDformer"],  # 15
                      # [EnhancedRNNModelNew(*param), "EnhancedRNNModelNew"],  # 16
                      # [TransformerModelNew(*param), "TransformerModelNew"],  # 17
                      # [EnhancedRNNModelV2New(*param), "EnhancedRNNModelV2New"]
                        ]

            for model, name in m_list:

                try:
                    train_model(model, name, train_losses_models)
                except Exception as e:
                    print(f"Ошибка при обучении модели {name}: {e}")
                model_path = os.path.join(MODEL_BASE, name, f"best_{name}.pt")  # final_model_
                model.load_state_dict(torch.load(model_path))
                model.eval()

                trade_game = TradeGame(model, x)
                val = sorted(trade_game.game(9999))
                for k in range(30):
                    test_data[k][0] += val[k][0]
            #     val = sorted(val)+[f"{name} {j} {i} {dropout}"]
            #     result.append(val)
            #
            #
            # for model_data in result:
            #     name = model_data[-1]  # Название модели
            #     values = model_data[:-1]  # Список значений
            #     print(f"{name}:")
            #     for i in range(0, len(values), 10):
            #         group = values[i:i + 10]
            #
            #         formatted_group = []
            #         for item in group:
            #             formatted_item = f"[{item[0]}, {int(item[2]):2d}]"
            #             formatted_group.append(formatted_item)
            #
            #         # Выводим группу из 5 элементов
            #         print("    " + " ".join(formatted_group))
            #     print()  # Пустая строка между моделями
            #
        for k in range(30):
            test_data[k][0] = round(test_data[k][0]/10, 1)
        print(f"{j} {i} {dropout}")
        print(test_data)
end_time = datetime.now()
end_ts = time.time()


# Запись в файл
with open("saved_models/time_log.txt", "a") as f:
    f.write(f"Начало: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Конец:  {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Затрачено времени: {end_ts - start_ts:.2f} секунд\n")
    f.write("=" * 30 + "\n")  # Разделитель между запусками

with open('train_losses_models.json', 'w') as f:
    json.dump(train_losses_models, f)

models_future(train_losses_models, current_train_samples=10, future_samples=100, epochs_to_consider=100)