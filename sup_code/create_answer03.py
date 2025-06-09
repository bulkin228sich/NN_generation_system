import os
import csv

# Пути
data_dir = r'C:\Users\dimon\PycharmProjects\dip\dataset\test\data'
answer1_dir = r'C:\Users\dimon\PycharmProjects\dip\dataset\test\answer'
output_dir = r'C:\Users\dimon\PycharmProjects\dip\dataset\test\answer03'

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
        close_prices = values[-3:]

        # Читаем значения из answer1 (предполагается, что в answer 3 строки с ответами)
        # Вместо использования csv.reader и next(reader) три раза
        with open(answer_path, 'r', encoding='utf-8') as f:
            # прочитать все непустые строки
            lines = [l.strip() for l in f.readlines() if l.strip()]
            # конвертировать в float
            answer_values = [float(l) for l in lines]

        # Открываем выходной файл для записи построчно
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Для каждого из трех выходных значений сравниваем и записываем результат
            for i in range(3):
                writer.writerow([0 if close_prices >= answer_values[i] else 1])

    except Exception as e:
        print(f"Ошибка обработки {filename}: {e}")