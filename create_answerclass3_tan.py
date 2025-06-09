import os
import csv

# Пути
data_dir = r'C:/Users/dimon/PycharmProjects/dip/dataset_new/training/data'
answer1_dir = r'C:/Users/dimon/PycharmProjects/dip/dataset_new/training/answer'
output_dir = r'C:/Users/dimon/PycharmProjects/dip/dataset_new/training/answerclass3_tan_full'

# Создать выходную папку, если её нет
os.makedirs(output_dir, exist_ok=True)
# Перебрать все CSV-файлы
for filename in os.listdir(data_dir):
    if not filename.endswith('.csv'):
        continue

    data_path = os.path.join(data_dir, filename)
    answer_path = os.path.join(answer1_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # Проверка: файл должен существовать в обеих папках
    if not os.path.exists(answer_path):
        print(f"Ответа нет для {filename}, пропущено")
        continue

    try:
        # Читаем 180-ю строку (индекс 179) из data
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) < 180:
                print(f"{filename}: слишком мало строк в данных")
                continue
            last_line = lines[179].strip()
            values = list(map(float, last_line.split(',')))

        # Получаем последние три значения
        close_prices = values[-3]

        # Открытие файла с ответами
        with open(answer_path, 'r', encoding='utf-8') as f:
            # Прочитать все непустые строки
            lines = [l.strip() for l in f.readlines() if l.strip()]
            # Конвертировать в float
            answer_values = [float(l) for l in lines]

        # Открытие выходного файла для записи
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Для каждого из трех выходных значений сравниваем и записываем результат
            for i in range(3):
                # Вычисление процентного изменения
                percent_change = ((answer_values[i] - close_prices) / close_prices) * 100

                # Округление до 2 знаков после запятой
                percent_change = round(percent_change, 2)*2

                # Если увеличение более 1%, записываем 1, иначе 0

                writer.writerow([percent_change])


    except Exception as e:
        print(f"Ошибка обработки {filename}: {e}")