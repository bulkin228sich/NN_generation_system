import os
import csv

# Исходная и целевая директории
input_dir = r"/dataset/test/data"
output_dir = r"/dataset/test/datashort"

# Создание целевой директории, если не существует
os.makedirs(output_dir, exist_ok=True)

# Обработка всех файлов в папке
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, newline='') as infile, open(output_path, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            for row in reader:
                # Оставляем только последние 3 значения
                if len(row) >= 3:
                    writer.writerow(row[-3:])