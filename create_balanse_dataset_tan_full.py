import os
import pandas as pd
import shutil
from collections import defaultdict

DATA_DIR = r'C:\Users\dimon\PycharmProjects\dip\dataset_new\test\data_procent_5'
ANSWER_DIR = r'C:\Users\dimon\PycharmProjects\dip\dataset_new\test\answerclass3_tan_full'

TRIMMED_DATA_DIR = r'C:\Users\dimon\PycharmProjects\dip\dataset_new\test\data_trimmed_tan_full'
TRIMMED_ANSWER_DIR = r'C:\Users\dimon\PycharmProjects\dip\dataset_new\test\answer_trimmed_tan_full'

os.makedirs(TRIMMED_DATA_DIR, exist_ok=True)
os.makedirs(TRIMMED_ANSWER_DIR, exist_ok=True)

# Подсчёт количества значений по диапазонам
buckets = defaultdict(list)  # ключ: индекс корзины 0..99, значение: список имён файлов

for filename in os.listdir(ANSWER_DIR):
    if not filename.endswith(".csv"):
        continue

    answer_path = os.path.join(ANSWER_DIR, filename)
    df = pd.read_csv(answer_path, header=None)

    # Предполагаем, что интересующее значение лежит в [0-строка, 0-й столбец]
    value = df.iloc[0, 0]

    # Вычисляем индекс корзины: сдвиг на +5, потом делим на 0.1 (т.е. умножаем на 10)
    # Если value == 5.0, поместим в верхнюю корзину (индекс 99)
    idx = int((value + 5.0) * 10)

    # Гарантируем, что idx попадает в диапазон [0..99]
    if idx < 0:
        idx = 0
    elif idx > 99:
        idx = 99

    buckets[idx].append(filename)

# Статистика по корзинам
for i in range(100):
    left = -5.0 + i * 0.1
    right = left + 0.1
    print(f"{left:.1f}–{right:.1f}: {len(buckets[i])} файлов")

# Сколько файлов выбрать из каждой корзины
TARGET_PER_BUCKET = int(input("Сколько файлов выбрать из каждой корзины: "))

# Центр диапазона: от -1.0 (включительно) до 2.0 (не включая).
# Индексы корзин, которые соответствуют этому интервалу:
#   left = -5.0 + i*0.1  → хотим -1.0 ≤ left < 2.0
#   i ≥ 40 (−1.0 = −5.0 + 40*0.1) и i < 70 (2.0 = −5.0 + 70*0.1)
CENTRAL_START = 30
CENTRAL_END   = 70  # не включая 70 → последние индексы: 40..69
for i in range(100):
    # Если корзина в центральном диапазоне, берём меньше файлов
    if CENTRAL_START <= i < CENTRAL_END:
        num_to_select = TARGET_PER_BUCKET // 5
    else:
        num_to_select = TARGET_PER_BUCKET

    # Срез при помощи [:num_to_select] автоматически берёт "сколько есть",
    # если элементов меньше, чем num_to_select.
    selected_files = buckets[i][:num_to_select]

    for fname in selected_files:
        data_path   = os.path.join(DATA_DIR, fname)
        answer_path = os.path.join(ANSWER_DIR, fname)

        shutil.copy2(data_path, os.path.join(TRIMMED_DATA_DIR, fname))
        shutil.copy2(answer_path, os.path.join(TRIMMED_ANSWER_DIR, fname))