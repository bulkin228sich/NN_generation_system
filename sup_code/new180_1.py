import os
import csv

# Пути к папкам
source_dir = r'C:\Users\dimon\PycharmProjects\dip\dataset\training\answer'
target_dir = r'C:\Users\dimon\PycharmProjects\dip\dataset\training\answer1'

# Создать целевую папку, если её нет
os.makedirs(target_dir, exist_ok=True)

# Перебрать все файлы в исходной папке
for filename in os.listdir(source_dir):
    if filename.endswith('.csv'):
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)

        # Чтение первой строки из исходного файла
        with open(source_path, 'r', newline='', encoding='utf-8') as src_file:
            reader = csv.reader(src_file)
            first_row = next(reader)  # Получить первую строку

        # Запись первой строки в новый файл
        with open(target_path, 'w', newline='', encoding='utf-8') as tgt_file:
            writer = csv.writer(tgt_file)
            writer.writerow(first_row)

source_dir = r'C:\Users\dimon\PycharmProjects\dip\dataset\test\answer'
target_dir = r'C:\Users\dimon\PycharmProjects\dip\dataset\test\answer1'

# Создать целевую папку, если её нет
os.makedirs(target_dir, exist_ok=True)

# Перебрать все файлы в исходной папке
for filename in os.listdir(source_dir):
    if filename.endswith('.csv'):
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)

        # Чтение первой строки из исходного файла
        with open(source_path, 'r', newline='', encoding='utf-8') as src_file:
            reader = csv.reader(src_file)
            first_row = next(reader)  # Получить первую строку

        # Запись первой строки в новый файл
        with open(target_path, 'w', newline='', encoding='utf-8') as tgt_file:
            writer = csv.writer(tgt_file)
            writer.writerow(first_row)