import os
import random
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# === НАСТРОЙКИ ===
INPUT_CSV = r'C:\Users\dimon\PycharmProjects\dip\sup_code\btc_10000.csv'
OUTPUT_DIR = r'C:\Users\dimon\PycharmProjects\dip\dataset_new'

WINDOW_SIZE = 180
PREDICT_OFFSETS = [0, 4, 9]  # через сколько шагов предсказывать
TEST_SPLIT_PROB = 0.2              # вероятность, что окно пойдет в тест
BUFFER_SIZE = 1000
MAX_WORKERS = os.cpu_count() * 2  # количество потоков

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===
def safe_filename(timestamp_str):
    return timestamp_str.replace(" ", "_").replace(":", "-")

def process_window(i, timestamps, features, closes):
    """Обработка одного окна"""
    try:
        ts = timestamps[i + WINDOW_SIZE - 1]
        window = features[i:i + WINDOW_SIZE]
        answer = [closes[i + WINDOW_SIZE + offset] for offset in PREDICT_OFFSETS]

        is_test = random.random() < TEST_SPLIT_PROB
        folder = 'test' if is_test else 'training'

        return ts, pd.DataFrame(window), pd.DataFrame([answer]), folder
    except Exception as e:
        print(f"Ошибка в обработке окна {i}: {e}")
        return None

def writer_worker(q, output_dir):
    """Поток записи файлов"""
    while True:
        item = q.get()
        if item is None:
            break

        ts, window, answer, folder = item
        base_path = Path(output_dir) / folder

        (base_path / 'data').mkdir(parents=True, exist_ok=True)
        (base_path / 'answer').mkdir(parents=True, exist_ok=True)

        window.to_csv(base_path / 'data' / f"{safe_filename(ts)}.csv", index=False, header=False)
        answer.T.to_csv(base_path / 'answer' / f"{safe_filename(ts)}.csv", index=False, header=False)

        q.task_done()

# === ОСНОВНАЯ ФУНКЦИЯ ===
def main():
    input_path = Path(INPUT_CSV)
    output_path = Path(OUTPUT_DIR)

    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"❌ Файл не найден: {INPUT_CSV}")
        return

    try:
        df = pd.read_csv(
            INPUT_CSV,
            header=None,
            dtype={
                0: 'object',  # Время
                1: np.float32,  # Open
                2: np.float32,  # High
                3: np.float32,  # Low
                4: np.float32,  # Close
                5: np.float32,  # Volume
                6: np.float32   # Доп. колонка
            }
        )

        timestamps = df[0].values.astype(str)  # Для имени файлов
        features = df.iloc[:, 1:].values.astype(np.float32)  # Только числовые данные
        closes = features[:, 3]  # Закрытия для ответов

        total_rows = len(features)
        max_offset = WINDOW_SIZE + max(PREDICT_OFFSETS)

        valid_indices = [
            i for i in range(total_rows - max_offset)
            if all((i + WINDOW_SIZE + offset) < total_rows for offset in PREDICT_OFFSETS)
        ]

        write_queue = queue.Queue(maxsize=10000)

        writer_thread = threading.Thread(
            target=writer_worker,
            args=(write_queue, OUTPUT_DIR),
            daemon=True
        )
        writer_thread.start()

        pbar = tqdm(total=len(valid_indices), desc="Обработка окон")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []

            for i in valid_indices:
                futures.append(executor.submit(process_window, i, timestamps, features, closes))

                if len(futures) >= MAX_WORKERS * 2:
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            write_queue.put(result)
                            pbar.update(1)
                    futures.clear()

            for future in as_completed(futures):
                result = future.result()
                if result:
                    write_queue.put(result)
                    pbar.update(1)

        write_queue.put(None)
        writer_thread.join()

        pbar.close()
        print(f"✅ Готово! Данные сохранены в {output_path}")

    except Exception as e:
        print(f"❌ Ошибка выполнения: {e}")

if __name__ == "__main__":
    main()
