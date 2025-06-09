import os
import csv

# Пути
data_dir = r'C:\Users\dimon\PycharmProjects\dip\dataset\test\data'
answer1_dir = r'C:\Users\dimon\PycharmProjects\dip\dataset\test\answer1'
output_dir = r'C:\Users\dimon\PycharmProjects\dip\dataset\test\answer0'

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

        # Получаем третье с конца значение
        close_price = values[-3]

        # Читаем значение из answer1
        with open(answer_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            answer_value = float(next(reader)[0])

        # Сравниваем
        label = 0 if close_price > answer_value else 1

        # Записываем результат
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([label])

    except Exception as e:
        print(f"Ошибка обработки {filename}: {e}")