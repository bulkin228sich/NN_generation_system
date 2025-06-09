import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


def load_data(data_dir, target_dir, percent=100):
    """
    Загружает данные и таргеты из файлов, с возможностью оставить определённый процент данных.
    :param data_dir: Путь к папке с данными
    :param target_dir: Путь к папке с целевыми метками
    :param percent: Процент данных для использования (от 0 до 100)
    :return: Списки данных и таргетов
    """
    data, targets = [], []

    # Получаем список файлов
    files = sorted(os.listdir(data_dir))

    # Количество данных, которое нужно оставить
    num_files = len(files)
    num_files_to_use = int(num_files * percent / 100)

    # Ограничиваем количество файлов
    files = files[:num_files_to_use]

    # Загружаем данные
    for file in files:
        x = np.loadtxt(os.path.join(data_dir, file), delimiter=',')  # (окно, признаки)
        y_path = os.path.join(target_dir, file)
        y = np.loadtxt(y_path, dtype=np.float32)[:1]


        targets.append(torch.tensor(y, dtype=torch.float32))

        data.append(torch.tensor(x, dtype=torch.float32))

    return data, targets


class PriceDataset(Dataset):
    def __init__(self, data, targets):
        assert len(data) == len(targets)
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

print("загрузка данных")
# Пример использования
train_data, train_targets = load_data("dataset_new/training/data_trimmed_tan_full", "dataset_new/training/answer_trimmed_tan_full",
                                      percent=100)  # Используем % данных
test_data, test_targets = load_data("dataset_new/test/data_trimmed_tan_full", "dataset_new/test/answer_trimmed_tan_full",
                                    percent=100)  # Используем  % данных

# Создание Dataset и DataLoader
train_dataset = PriceDataset(train_data, train_targets)
test_dataset = PriceDataset(test_data, test_targets)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
val_loader = DataLoader(test_dataset, batch_size=16, pin_memory=True)

