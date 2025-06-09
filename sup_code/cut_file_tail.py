import pandas as pd

input_file  = r'C:\Users\dimon\PycharmProjects\dip\sup_code\btc_100000.csv'
output_file = 'btc_10000.csv'
x = 10000

# 1) Читаем файл без собственного заголовка
df = pd.read_csv(input_file, header=None)

# 2) Оставляем последние x строк
df = df.tail(x).reset_index(drop=True)

# 3) Удаляем столбец с датой/временем (он в нулевом столбце)
#df = df.drop(columns=[0])

# 4) Сохраняем без заголовка и без индекса
df.to_csv(output_file, header=False, index=False)

print(f"Оставлены последние {x} строк без столбца даты/времени. Результат в {output_file}")