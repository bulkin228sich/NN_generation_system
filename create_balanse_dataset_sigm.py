import os
import pandas as pd
import shutil
from collections import defaultdict

DATA_DIR = r'C:\Users\dimon\PycharmProjects\dip\dataset_new\training\data_procent_5'
ANSWER_DIR = r'C:\Users\dimon\PycharmProjects\dip\dataset_new\training\answerclass3_sigm'

TRIMMED_DATA_DIR = r'C:\Users\dimon\PycharmProjects\dip\dataset_new\training\data_trimmed'
TRIMMED_ANSWER_DIR = r'C:\Users\dimon\PycharmProjects\dip\dataset_new\training\answer_trimmed_sigm'

os.makedirs(TRIMMED_DATA_DIR, exist_ok=True)
os.makedirs(TRIMMED_ANSWER_DIR, exist_ok=True)

# Подсчёт количества значений по диапазонам
buckets = defaultdict(list)  # ключ: диапазон (0.0–0.1 и т.д.), значение: список имён файлов

for filename in os.listdir(ANSWER_DIR):
    if filename.endswith(".csv"):
        answer_path = os.path.join(ANSWER_DIR, filename)
        df = pd.read_csv(answer_path, header=None)

        value = df.iloc[0, 0]
        bucket_index = min(int(value * 10), 9)  # 0–0.1 -> 0, ..., 0.9–1.0 -> 9
        buckets[bucket_index].append(filename)

# Статистика по корзинам
for i in range(10):
    print(f"{i/10:.1f}–{(i+1)/10:.1f}: {len(buckets[i])} файлов")

# Число файлов, которые нужно выбрать из каждого диапазона
TARGET_PER_BUCKET = int(input())

# Создание урезанного датасета
for i in range(10):
    if i < 4 or i > 6:
        selected_files = buckets[i][:TARGET_PER_BUCKET]
    else:
        selected_files = buckets[i][:int(TARGET_PER_BUCKET/2)]
    for file in selected_files:
        data_path = os.path.join(DATA_DIR, file)
        answer_path = os.path.join(ANSWER_DIR, file)
        shutil.copy2(data_path, os.path.join(TRIMMED_DATA_DIR, file))
        shutil.copy2(answer_path, os.path.join(TRIMMED_ANSWER_DIR, file))