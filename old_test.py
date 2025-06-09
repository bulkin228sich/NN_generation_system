import tkinter as tk
from tkinter import ttk, messagebox

class ModelSearchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hyperparameter Search Interface")
        self.geometry("900x700")
        self.create_widgets()

    def create_widgets(self):
        # -------------------------------
        # 1. MODEL SELECTION FRAME
        # -------------------------------
        model_frame = ttk.LabelFrame(self, text="Model Architecture")
        model_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(model_frame, text="Select Model:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_var = tk.StringVar()
        model_options = [
            "LinearModel", "MLPModel", "GRUModel", "EnhancedRNNModel",
            "TransformerModel", "SimpleInformer", "TimesNetClassifier",
            "GRUD", "SCINet", "TransformerModelNew", "EnhancedRNNModelV2New"
        ]
        model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=model_options,
            state="readonly",
            width=25
        )
        model_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        model_combo.current(0)

        # -------------------------------
        # 2. HYPERPARAMETERS FRAME
        # -------------------------------
        hyper_frame = ttk.LabelFrame(self, text="Hyperparameters Grid Search")
        hyper_frame.pack(fill="x", padx=10, pady=5)

        # Словарь для хранения настроек гиперпараметров
        self.hyper_params = {
            "learning_rate":   {"var": tk.BooleanVar(value=False), "entry": None},
            "weight_decay":    {"var": tk.BooleanVar(value=False), "entry": None},
            "batch_size":      {"var": tk.BooleanVar(value=False), "entry": None},
            "hidden_size":     {"var": tk.BooleanVar(value=False), "entry": None},
            "num_layers":      {"var": tk.BooleanVar(value=False), "entry": None},
            "dropout_rate":    {"var": tk.BooleanVar(value=False), "entry": None},
        }

        for idx, (param, data) in enumerate(self.hyper_params.items()):
            # Чекбокс, включить/выключить перебор этого параметра
            chk = ttk.Checkbutton(hyper_frame, text=param, variable=data["var"])
            chk.grid(row=idx, column=0, padx=5, pady=2, sticky="w")
            # Ввод значений через запятую или как min,max,step
            entry = ttk.Entry(hyper_frame, width=40)
            entry.grid(row=idx, column=1, padx=5, pady=2)
            entry.insert(0, "e.g. 0.0001,0.0005,0.001 or min,max,step")
            data["entry"] = entry

        calc_btn = ttk.Button(
            hyper_frame,
            text="Calculate Iterations",
            command=self.calculate_iterations
        )
        calc_btn.grid(row=len(self.hyper_params), column=0, padx=5, pady=10, sticky="w")

        # -------------------------------
        # 3. DATA PREPARATION FRAME
        # -------------------------------
        data_frame = ttk.LabelFrame(self, text="Data Preparation")
        data_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(data_frame, text="Percent of Data:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.percent_entry = ttk.Entry(data_frame, width=10)
        self.percent_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.percent_entry.insert(0, "1.0")

        ttk.Label(data_frame, text="Window Size:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.window_entry = ttk.Entry(data_frame, width=10)
        self.window_entry.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.window_entry.insert(0, "120")

        ttk.Label(data_frame, text="Horizon:").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.horizon_entry = ttk.Entry(data_frame, width=10)
        self.horizon_entry.grid(row=0, column=5, padx=5, pady=5, sticky="w")
        self.horizon_entry.insert(0, "1")

        # -------------------------------
        # 4. METRICS AND VALIDATION FRAME
        # -------------------------------
        metrics_frame = ttk.LabelFrame(self, text="Metrics and Validation")
        metrics_frame.pack(fill="x", padx=10, pady=5)

        self.metrics_vars = {
            "MSE":      tk.BooleanVar(),
            "MAE":      tk.BooleanVar(),
            "Accuracy": tk.BooleanVar(),
            "F1-score": tk.BooleanVar(),
            "ROC-AUC":  tk.BooleanVar(),
        }
        for idx, (metric, var) in enumerate(self.metrics_vars.items()):
            chk = ttk.Checkbutton(metrics_frame, text=metric, variable=var)
            chk.grid(row=0, column=idx, padx=5, pady=5, sticky="w")

        # -------------------------------
        # 5. COMPUTE SETTINGS FRAME
        # -------------------------------
        compute_frame = ttk.LabelFrame(self, text="Compute Settings")
        compute_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(compute_frame, text="Device:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.device_var = tk.StringVar(value="cuda")
        device_combo = ttk.Combobox(
            compute_frame,
            textvariable=self.device_var,
            values=["cuda", "cpu"],
            state="readonly",
            width=7
        )
        device_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(compute_frame, text="Num Workers:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.workers_entry = ttk.Entry(compute_frame, width=5)
        self.workers_entry.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.workers_entry.insert(0, "4")

        ttk.Label(compute_frame, text="Pin Memory:").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.pinmem_var = tk.BooleanVar(value=True)
        pinmem_chk = ttk.Checkbutton(compute_frame, variable=self.pinmem_var)
        pinmem_chk.grid(row=0, column=5, padx=5, pady=5, sticky="w")

        # -------------------------------
        # 6. CONTROL BUTTONS FRAME
        # -------------------------------
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=10, pady=10)

        start_btn = ttk.Button(btn_frame, text="Start", command=self.start_search)
        start_btn.pack(side="left", padx=5)

        stop_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_search)
        stop_btn.pack(side="left", padx=5)

        reset_btn = ttk.Button(btn_frame, text="Reset", command=self.reset_all)
        reset_btn.pack(side="left", padx=5)

        # -------------------------------
        # 7. STATUS & PROGRESS FRAME
        # -------------------------------
        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x", padx=10, pady=5)

        self.progress = ttk.Progressbar(
            status_frame,
            orient="horizontal",
            length=600,
            mode="determinate"
        )
        self.progress.pack(side="left", padx=5, pady=5)

        self.status_label = ttk.Label(status_frame, text="Iterations: 0 | ETA: --:--:--")
        self.status_label.pack(side="left", padx=10)

    # Заглушка для расчёта числа итераций
    def calculate_iterations(self):
        messagebox.showinfo("Info", "Calculating total iterations...")

    # Заглушка для начала поиска
    def start_search(self):
        messagebox.showinfo("Info", "Starting hyperparameter search...")

    # Заглушка для остановки
    def stop_search(self):
        messagebox.showinfo("Info", "Stopping search...")

    # Сброс всех полей к значениям по умолчанию
    def reset_all(self):
        self.model_var.set("")
        for data in self.hyper_params.values():
            data["var"].set(False)
            data["entry"].delete(0, tk.END)
            data["entry"].insert(0, "e.g. 0.0001,0.0005,0.001 or min,max,step")
        for var in self.metrics_vars.values():
            var.set(False)
        self.percent_entry.delete(0, tk.END)
        self.percent_entry.insert(0, "1.0")
        self.window_entry.delete(0, tk.END)
        self.window_entry.insert(0, "120")
        self.horizon_entry.delete(0, tk.END)
        self.horizon_entry.insert(0, "1")
        self.device_var.set("cuda")
        self.workers_entry.delete(0, tk.END)
        self.workers_entry.insert(0, "4")
        self.pinmem_var.set(True)
        self.progress['value'] = 0
        self.status_label.config(text="Iterations: 0 | ETA: --:--:--")
        messagebox.showinfo("Info", "All fields reset to default.")

if __name__ == "__main__":
    app = ModelSearchApp()
    app.mainloop()