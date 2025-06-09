import time
import torch
import torch.nn as nn
import os
from datetime import datetime
from tqdm import tqdm
from load_dataset import *
from schedule import models_train
class EarlyStopping:
    def __init__(self, patience=15, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# ===== 6. Тренировка модели =====
def train_model(model, model_name, train_losses_models,  epochs=100, lr=0.0001):
    early_stopper = EarlyStopping(patience=15)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # Создание папки для модели
    save_dir = os.path.join("saved_models", model_name)
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    # Логирование
    log_path = os.path.join(save_dir, "training_log.txt")
    with open(log_path, "w") as log_file:
        log_file.write(f"Training log for model: {model_name}\n")
        log_file.write(f"Started at: {datetime.now()}\n\n")

    # Общее время обучения
    total_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            pared = model(x)
            loss = criterion(pared, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pared = model(x)
                val_loss += criterion(pared, y).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        epoch_time = time.time() - epoch_start_time

        # log_message = (f"Epoch {epoch + 1}/{epochs} | "
        #                f"Train Loss: {avg_train_loss:.6f} | "
        #                f"Val Loss: {avg_val_loss:.6f} | "
        #                f"Time: {epoch_time:.2f} sec")
        time.sleep(0.01)
        #tqdm.write(log_message)
        time.sleep(0.01)
        #with open(log_path, "a") as log_file:
            #log_file.write(log_message + "\n")

        # Сохраняем лучшую модель
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(save_dir, f"best_{model_name}.pt")
            torch.save(model.state_dict(), best_model_path)
            #print(f"\n💾 best model saved: {avg_val_loss} {epoch+1}")

        # Сохраняем модель эпохи
        epoch_model_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_model_path)

        early_stopper.step(avg_val_loss)
        if early_stopper.should_stop:
            #print("🛑 Early stopping triggered!")
            break

    total_training_time = time.time() - total_start_time

    # # Расчет прогнозируемого loss на 2 000 000 данных
    # current_train_samples = 100000
    # future_samples = 2000000
    # if len(train_losses) >= 2:
    #     # Последние две эпохи
    #     loss1 = train_losses[-2]
    #     loss2 = train_losses[-1]
    #
    #     # Проверим, чтобы значения сэмплов были разные
    #     earlier_samples = current_train_samples // 2  # допустим, предположим
    #     current_samples = current_train_samples
    #
    #     if current_samples == earlier_samples:
    #         k = 0.1  # если нет разницы, фиксируем маленький прогресс
    #     else:
    #         # Вычисляем коэффициент скорости снижения лосса
    #         k = np.log(loss1 / loss2) / (np.log(current_samples / earlier_samples + 1e-8))
    #
    #     # Теперь прогноз на будущем количестве данных
    #     predicted_loss = train_losses[-1] * (current_samples / future_samples) ** k
    #
    #     with open(log_path, "a") as log_file:
    #         log_file.write(f"\nPrediction:\n")
    #         log_file.write(f"Estimated Train Loss at {future_samples} samples: {predicted_loss:.8f}\n")
    #         predicted_loss = round(predicted_loss, 3)
    #         print(f"Estimated Train Loss at {future_samples} samples: {predicted_loss:.8f}\n")
    # else:
    #     with open(log_path, "a") as log_file:
    #         log_file.write("\nNot enough epochs to estimate future loss.\n")
    #
    # with open(log_path, "a") as log_file:
    #     log_file.write(f"\nTotal training time: {total_training_time:.2f} sec\n")
    #     print(f"\nTotal training time: {total_training_time:.2f} sec\n")
    #     log_file.write(f"Finished at: {datetime.now()}\n")
    #
    # #График
    # models_train(train_losses, val_losses, model_name, save_dir)
    #
    # train_losses_models.append([train_losses, f"{model_name}"])

