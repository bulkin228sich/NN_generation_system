import os
import pandas as pd
import torch
import numpy as np
from torch import nn
import math
from models import *

m_list = [[LinearModel(), "LinearModel"],  # 0
          [MLPModel(), "MLPModel"],  # 1
          [GRUModel(), "GRUModel"],  # 2
          [EnhancedRNNModel(), "EnhancedRNNModel"],  # 3
          [TransformerModel(), "TransformerModel"],  # 4
          [EnhancedRNNModelV2(), "EnhancedRNNModelV2"],  # 5
          [TCNModel(), "TCNModel"],  # 6
          [NBeatsModel(), "NBeatsModel"],  # 7
          [SimpleInformer(), "SimpleInformer"],  # 8
          [TFTLite(), "TFTLite"],  # 9
          [SimpleDeepAR(), "SimpleDeepAR"],  # 10
          [PatchTST(), "PatchTST"],  # 11
          [TimesNet(), "TimesNet"],  # 12
          [GRUD(), "GRUD"],  # 13
          [SCINet(), "SCINet"],  # 14
          [FEDformer(), "FEDformer"]]
# Функция загрузки модели
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Переводим модель в режим инференса
    return model


# Преобразуем строку данных в формат, подходящий для модели
def process_input(multiline_string):
    # Разбиваем по строкам
    lines = multiline_string.strip().splitlines()

    # Каждую строку разбиваем по запятой, убираем первый столбец (дату), и преобразуем к float
    data = []
    for line in lines:
        parts = line.strip().split(',')[1:]  # убираем дату
        float_parts = [float(p) for p in parts]
        data.extend(float_parts)  # собираем в один список

    # Преобразуем в numpy-массив и тензор
    input_data = np.array(data).reshape(1, -1)  # Подготовка данных в форму (1, 120)
    return torch.tensor(input_data, dtype=torch.float32)


# Прогнозируем с помощью модели
def predict(model, data):
    with torch.no_grad():  # Отключаем вычисление градиентов
        prediction = model(data)
    return prediction


# Основная функция
def main():
    model_list = [[LinearModel(), "LinearModel"],  # 0
          [MLPModel(), "MLPModel"],  # 1
          [GRUModel(), "GRUModel"],  # 2
          [EnhancedRNNModel(), "EnhancedRNNModel"],  # 3
          [TransformerModel(), "TransformerModel"],  # 4
          [EnhancedRNNModelV2(), "EnhancedRNNModelV2"],  # 5
          [TCNModel(), "TCNModel"],  # 6
          [NBeatsModel(), "NBeatsModel"],  # 7
          [SimpleInformer(), "SimpleInformer"],  # 8
          [TFTLite(), "TFTLite"],  # 9
          [SimpleDeepAR(), "SimpleDeepAR"],  # 10
          [PatchTST(), "PatchTST"],  # 11
          [TimesNet(), "TimesNet"],  # 12
          [GRUD(), "GRUD"],  # 13
          [SCINet(), "SCINet"],  # 14
          [FEDformer(), "FEDformer"]]




    DATA_DIR = r'C:\Users\dimon\Desktop\Диплом\Модели и код\20_100000\dataset\test\data'
    ANSWER_DIR = r'C:\Users\dimon\Desktop\Диплом\Модели и код\20_100000\dataset\test\answer1'
    data_files = sorted(os.listdir(DATA_DIR))
    answer_files = sorted(os.listdir(ANSWER_DIR))
    answers = []
    for fn in answer_files:
        path = os.path.join(ANSWER_DIR, fn)
        arr = pd.read_csv(path, header=None).values.flatten().astype(int)  # (3,)
        answers.append(arr.tolist())

    for model, name in m_list:
        # 1) Загрузить веса
        try:

            model_path = os.path.join(f"saved_models", name, f'best_{name}.pt')  # final_model_
            model.load_state_dict(torch.load(model_path))
            model.eval()

            # 2) Счётчики совпадений: один слот на каждый из трёх интервалов
            counts = [name, 0]

            # 3) Пройти по 10 (или len(data_files)) примерам
            for idx, fn in enumerate(data_files[:1000]):
                # 3.1) Загрузить данные
                df = pd.read_csv(os.path.join(DATA_DIR, fn), header=None).values  # (180,6)
                x = torch.tensor(df, dtype=torch.float32).unsqueeze(0)  # (1,180,6)

                # 3.2) Предсказание
                with torch.no_grad():
                    out = model(x)  # (1,3)
                    pred = torch.round(out[0]).int().tolist()  # [p1,p2,p3]

                # 3.3) Сравнить с ответом
                y_true = answers[idx]  # [t1,t2,t3]
                counts[1] += abs(abs(pred[0]) - abs(y_true[0]))
                #counts[2] += abs(abs(pred[1]) - abs(y_true[1]))
                #counts[3] += abs(abs(pred[2]) - abs(y_true[2]))
                # 5) Вывести
            print(f" {counts[1]/1000}   ")
        except Exception as e:

            print(f"Ошибка при обучении модели {name}: {e}")



if __name__ == "__main__":
    main()