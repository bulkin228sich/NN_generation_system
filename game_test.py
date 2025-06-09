
import numpy as np

import os
import torch
import torch.nn as nn
import pandas as pd
from models import *
from trade_game import TradeGame


data_file = "btc_10180.csv"
x = np.loadtxt(os.path.join("sup_code", data_file), delimiter=',')
MODEL_BASE = r'saved_models'
result = []
seq_len = 180
input_size = 5
out_d = 1
num_layers=1
hidden_size =32
dropout = 0.1
param = [seq_len, input_size,  out_d, num_layers, hidden_size, dropout]
m_list = [
            #(LinearModel(*param), "LinearModel"),
           #(MLPModel(*param), "MLPModel"),
           [GRUModel(*param), "GRUModel"],  # 2
    #       [EnhancedRNNModel(*param), "EnhancedRNNModel"],  # 3
    #       [TransformerModel(*param), "TransformerModel"],  # 4
    #       [EnhancedRNNModelV2(*param), "EnhancedRNNModelV2"],  # 5
    #       [TCNModel(*param), "TCNModel"],  # 6
    #       [NBeatsModel(*param), "NBeatsModel"],  # 7
    #       [SimpleInformer(*param), "SimpleInformer"],  # 8
    #       [TFTLite(*param), "TFTLite"],  # 9
    #       [SimpleDeepAR(*param), "SimpleDeepAR"],  # 10
    #       [PatchTST(*param), "PatchTST"],  # 11
    #       [TimesNetClassifier(*param), "TimesNetClassifier"],  # 12
    #       [GRUD(*param), "GRUD"],  # 13
    #       [SCINet(*param), "SCINet"],  # 14
    #       [FEDformer(*param), "FEDformer"],  # 15
    #       [EnhancedRNNModelNew(*param), "EnhancedRNNModelNew"],  # 16
    #       [TransformerModelNew(*param), "TransformerModelNew"],  # 17
    #       [EnhancedRNNModelV2New(*param), "EnhancedRNNModelV2New"]
            ]
for model, name in m_list:

    model_path = os.path.join(MODEL_BASE, name, f"best_{name}.pt")  # final_model_   model_epoch_14  best_{name}
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(torch.device("cuda"))
    print(name)
    trade_game = TradeGame(model, x)
    val = sorted(trade_game.game(9999))+[f"{name} {num_layers} {hidden_size} {dropout}"]
    result.append(val)

for model_data in result:
    name = model_data[-1]  # Название модели
    values = model_data[:-1]  # Список значений
    print(f"{name}:")
    for i in range(0, len(values), 10):
        group = values[i:i + 10]

        formatted_group = []
        for item in group:
            formatted_item = f"[{item[0]}, {int(item[2]):2d}]"
            formatted_group.append(formatted_item)

        # Выводим группу из 5 элементов
        print("    " + " ".join(formatted_group))

    print()  # Пустая строка между моделями