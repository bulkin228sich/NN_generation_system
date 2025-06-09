import os
import torch
import torch.nn as nn
import pandas as pd
from models import *
# Папки
DATA_DIR = r'C:\Users\dimon\PycharmProjects\dip\dataset_new\training\data_trimmed'
ANSWER_DIR = r'C:\Users\dimon\PycharmProjects\dip\dataset_new\training\answer_trimmed'
MODEL_BASE = r'saved_models'

# Список моделей
 # 18


# Собираем имена файлов и ожидаемые ответы
data_files = sorted(os.listdir(DATA_DIR))
answer_files = sorted(os.listdir(ANSWER_DIR))
# Предполагаем, что их одинаково много и по порядку

# Загрузим все ответы сразу: answers[i] = [y1, y2, y3]
answers = []
for fn in answer_files:
    path = os.path.join(ANSWER_DIR, fn)
    arr = pd.read_csv(path, header=None).values.flatten().astype(float)  # (3,)
    answers.append(arr.tolist())
arr = []
# Проходим по моделям
for model, name in m_list:
    # 1) Загрузить веса


        model_path = os.path.join(MODEL_BASE, name, f'best_{name}.pt') #final_model_
        model.load_state_dict(torch.load(model_path))
        model.eval()
        counts = [name, 0, 0, 0]
        # 2) Счётчики совпадений: один слот на каждый из трёх интервалов
        val = [float('inf'), float('-inf')]  # [min, max]

        # Сначала собираем min/max
        # for idx, fn in enumerate(data_files[:1000]):
        #     df = pd.read_csv(os.path.join(DATA_DIR, fn), header=None).values  # (180,6)
        #     x = torch.tensor(df, dtype=torch.float32).unsqueeze(0)  # (1,180,6)
        #
        #     with torch.no_grad():
        #         out = model(x).tolist()[0]  # [p1, p2, p3]
        #         val[0] = min(val[0], out[0])
        #         val[1] = max(val[1], out[0])
        #
        # # Пример нормализации нового выхода:
        # # normalized = (value - min_val) / (max_val - min_val)
        # # Защита от деления на ноль
        # min_val, max_val = val
        # range_val = max_val - min_val if max_val != min_val else 1e-6

        # 3) Пройти по 10 (или len(data_files)) примерам
        for idx, fn in enumerate(data_files[:1000]):
            # 3.1) Загрузить данные
            df = pd.read_csv(os.path.join(DATA_DIR, fn), header=None).values  # (180,6)
            x = torch.tensor(df, dtype=torch.float32).unsqueeze(0)  # (1,180,6)

            # 3.2) Предсказание
            with torch.no_grad():
                out = model(x).tolist()           # (1,3)
                out = out[0]  # [p1,p2,p3]
                # out = (out[0] - min_val) / range_val
            # 3.3) Сравнить с ответом
            y_true = answers[idx]  # [t1,t2,t3]
            for j in range(1):
                x = abs(out[j] - y_true[j])
                if x < 0.3:
                    counts[j+1] += 1

        # 5) Вывести
        print(f"{counts[0]} 1min: {counts[1]/1}%  5min: {counts[2]/10}%  10min: {counts[3]/10}%")
        arr.append(counts)

for i in arr:
    print(i[1]/10)
print("///")
for i in arr:
    print(i[2]/10)
print("///")
for i in arr:
    print(i[3]/10)