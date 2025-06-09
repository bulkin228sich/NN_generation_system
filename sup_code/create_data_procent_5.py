import os
import pandas as pd

# Путь к папке с исходными данными
input_dir = r'C:\Users\dimon\PycharmProjects\dip\dataset_new\test\data'
# Путь к папке для сохранения новых файлов
output_dir = os.path.join(input_dir + '_procent_5')
os.makedirs(output_dir, exist_ok=True)

# Обход всех CSV-файлов
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        path = os.path.join(input_dir, filename)
        df = pd.read_csv(path, header=None)

        # Переименование столбцов для удобства
        df.columns = ['Open', 'High', 'Low', 'Close', 'VolumeBTC', 'VolumeUSD']

        # Расчёт процентных изменений
        df['HighPct'] = (df['High'] - df['Open']) / df['Open'] * 100
        df['LowPct'] = (df['Low'] - df['Open']) / df['Open'] * 100
        df['ClosePct'] = (df['Close'] - df['Open']) / df['Open'] * 100

        # Масштабирование объёма в долларах
        df['VolumeUSD'] = df['VolumeUSD'] / 1_000_000

        # Выбор нужных столбцов и порядок
        new_df = df[['HighPct', 'LowPct', 'ClosePct', 'VolumeBTC', 'VolumeUSD']]

        # Округление до 4 знаков после запятой
        new_df = new_df.round(4)

        # Сохранение результата
        new_path = os.path.join(output_dir, filename)
        new_df.to_csv(new_path, index=False, header=False)