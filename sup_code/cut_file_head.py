import pandas as pd

input_file  = r'C:\Users\dimon\PycharmProjects\dip\sup_code\btc_1m_from_2025-04-14.csv'
output_file = 'btc_10180.csv'
x = 10000

# 1) Читаем файл без собственного заголовка, пропуская первую строку исходного файла
df = pd.read_csv(input_file, header=None)

# 2) Оставляем первые x строк
df = df.head(x).reset_index(drop=True)

# 3) Удаляем столбец с датой/временем (он в нулевом столбце)
df = df.drop(columns=[0])

# 4) Сохраняем без заголовка и без индекса
df.to_csv(output_file, header=False, index=False)

print(f"Оставлены первые {x} строк без столбца даты/времени. Результат в {output_file}")

