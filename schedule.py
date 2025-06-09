import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

def models_train(train_losses, val_losses, model_name, save_dir):
    # График
    fig, ax = plt.subplots()
    ax.set_yscale('symlog')

    def y_fmt(x, pos):
        return f'{x / 1e6:.0f}M' if x >= 1e6 else f'{x:.0f}'

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))
    ax.set_ylabel('Loss (log scale)')

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"{model_name}")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    loss_curve_path = os.path.join(save_dir, f"{model_name}.png")
    plt.savefig(loss_curve_path)



def models_future(train_losses_models, current_train_samples = 10000, future_samples = 2000000, epochs_to_consider = 20 ):
    # Готовим график
    plt.figure(figsize=(10, 6))
    plt.title(f"Прогноз функции потерь до {future_samples:,} обучающих примеров")
    plt.xlabel("Количество обучающих примеров")
    plt.ylabel("Ожидаемый Loss")
    plt.grid(True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(train_losses_models)))

    # Проходим по всем моделям
    for idx, (losses, model_name) in enumerate(train_losses_models):
        # Берем только последние epochs_to_consider эпох
        if len(losses) < 2:
            continue  # мало данных для прогноза

        recent_losses = losses[-epochs_to_consider:]
        loss1 = recent_losses[-2]
        loss2 = recent_losses[-1]

        # Для расчета скорости уменьшения
        earlier_samples = current_train_samples - (current_train_samples // epochs_to_consider)
        current_samples = current_train_samples

        if current_samples == earlier_samples:
            k = 0.1
        else:
            k = np.log(loss1 / loss2) / (np.log(current_samples / earlier_samples + 1e-8))

        # Подготовка точек для прогноза
        sample_points = np.logspace(np.log10(current_train_samples), np.log10(future_samples), 100)
        predicted_losses = recent_losses[-1] * (current_train_samples / sample_points) ** k

        # Рисуем линию прогноза
        plt.plot(sample_points, predicted_losses, label=f"{model_name} (k={k:.2f})", color=colors[idx])

    # Финальный штрих
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    save_dir = os.path.join("saved_models")
    plt.savefig(save_dir)
    plt.show()

