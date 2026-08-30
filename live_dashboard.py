import tkinter as tk
from tkinter import ttk
import random
import math


class PredictiveMaintenanceDashboard:

    def __init__(self):
        self.root = tk.Tk()

        self.root.title(
            "Edge AI Predictive Maintenance System"
        )

        self.root.geometry("1250x760")
        self.root.minsize(1100, 680)

        self.root.configure(
            bg="#0b1220"
        )

        self.running = False
        self.step = 0

        # --------------------------------------------------
        # FINAL RESULTS FROM YOUR CURRENT PIPELINE
        # --------------------------------------------------

        self.isolation_accuracy = 81.29
        self.isolation_f1 = 61.20
        self.isolation_auc = 91.23

        self.autoencoder_accuracy = 91.97
        self.autoencoder_f1 = 76.74
        self.autoencoder_auc = 95.76

        self.fusion_accuracy = 90.45
        self.fusion_f1 = 70.15
        self.fusion_auc = 92.71

        self.rul_mae = 21.07
        self.rul_rmse = 34.53
        self.rul_r2 = 0.6648

        self.build_interface()

    # ======================================================
    # MAIN INTERFACE
    # ======================================================

    def build_interface(self):

        # --------------------------------------------------
        # HEADER
        # --------------------------------------------------

        header = tk.Frame(
            self.root,
            bg="#111c32",
            height=90
        )

        header.pack(
            fill="x",
            padx=18,
            pady=(18, 10)
        )

        tk.Label(
            header,
            text="EDGE AI",
            font=("Segoe UI", 13, "bold"),
            fg="#55d6ff",
            bg="#111c32"
        ).pack(
            anchor="w",
            padx=25,
            pady=(16, 0)
        )

        tk.Label(
            header,
            text="Predictive Maintenance & Fault Detection",
            font=("Segoe UI", 25, "bold"),
            fg="white",
            bg="#111c32"
        ).pack(
            anchor="w",
            padx=25
        )

        self.status_label = tk.Label(
            header,
            text="● SYSTEM READY",
            font=("Segoe UI", 12, "bold"),
            fg="#42e88b",
            bg="#111c32"
        )

        self.status_label.place(
            relx=0.97,
            rely=0.5,
            anchor="e"
        )

        # --------------------------------------------------
        # CONTENT
        # --------------------------------------------------

        content = tk.Frame(
            self.root,
            bg="#0b1220"
        )

        content.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=5
        )

        # LEFT SIDE
        left = tk.Frame(
            content,
            bg="#0b1220"
        )

        left.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 8)
        )

        # RIGHT SIDE
        right = tk.Frame(
            content,
            bg="#0b1220",
            width=380
        )

        right.pack(
            side="right",
            fill="y",
            padx=(8, 0)
        )

        # ==================================================
        # LIVE MACHINE STATUS
        # ==================================================

        self.create_section_title(
            left,
            "LIVE MACHINE MONITOR"
        )

        machine = tk.Frame(
            left,
            bg="#111c32",
            height=125
        )

        machine.pack(
            fill="x",
            pady=(0, 12)
        )

        self.machine_status = tk.Label(
            machine,
            text="● NORMAL",
            font=("Segoe UI", 28, "bold"),
            fg="#42e88b",
            bg="#111c32"
        )

        self.machine_status.pack(
            pady=(20, 2)
        )

        self.machine_message = tk.Label(
            machine,
            text="Machine operating within expected range",
            font=("Segoe UI", 11),
            fg="#aab8d0",
            bg="#111c32"
        )

        self.machine_message.pack()

        # ==================================================
        # SENSOR MONITOR
        # ==================================================

        self.create_section_title(
            left,
            "SENSOR STREAM"
        )

        sensor_frame = tk.Frame(
            left,
            bg="#111c32"
        )

        sensor_frame.pack(
            fill="x",
            pady=(0, 12)
        )

        self.sensor_bars = {}

        sensors = [
            "Temperature",
            "Pressure",
            "Speed",
            "Vibration",
            "Flow"
        ]

        for sensor in sensors:

            row = tk.Frame(
                sensor_frame,
                bg="#111c32"
            )

            row.pack(
                fill="x",
                padx=20,
                pady=7
            )

            tk.Label(
                row,
                text=sensor,
                width=14,
                anchor="w",
                font=("Segoe UI", 10),
                fg="#d7e1f2",
                bg="#111c32"
            ).pack(
                side="left"
            )

            bar = ttk.Progressbar(
                row,
                length=300,
                mode="determinate"
            )

            bar.pack(
                side="left",
                padx=10
            )

            value = tk.Label(
                row,
                text="50%",
                width=7,
                font=("Segoe UI", 10, "bold"),
                fg="#55d6ff",
                bg="#111c32"
            )

            value.pack(
                side="left"
            )

            self.sensor_bars[sensor] = (
                bar,
                value
            )

        # ==================================================
        # MODEL STATUS
        # ==================================================

        self.create_section_title(
            left,
            "FAULT DETECTION MODELS"
        )

        model_frame = tk.Frame(
            left,
            bg="#111c32"
        )

        model_frame.pack(
            fill="x"
        )

        self.create_model_row(
            model_frame,
            "Isolation Forest",
            self.isolation_accuracy,
            self.isolation_f1,
            self.isolation_auc
        )

        self.create_model_row(
            model_frame,
            "Dense Autoencoder",
            self.autoencoder_accuracy,
            self.autoencoder_f1,
            self.autoencoder_auc
        )

        self.create_model_row(
            model_frame,
            "Proposed Fusion",
            self.fusion_accuracy,
            self.fusion_f1,
            self.fusion_auc
        )

        # ==================================================
        # RIGHT PANEL
        # ==================================================

        self.create_section_title(
            right,
            "RUL PREDICTION"
        )

        rul_frame = tk.Frame(
            right,
            bg="#111c32"
        )

        rul_frame.pack(
            fill="x",
            pady=(0, 15)
        )

        self.create_big_metric(
            rul_frame,
            "LSTM R²",
            f"{self.rul_r2:.4f}"
        )

        self.create_big_metric(
            rul_frame,
            "MAE",
            f"{self.rul_mae:.2f} cycles"
        )

        self.create_big_metric(
            rul_frame,
            "RMSE",
            f"{self.rul_rmse:.2f} cycles"
        )

        # ==================================================
        # DATASET
        # ==================================================

        self.create_section_title(
            right,
            "EXPERIMENTAL SETUP"
        )

        setup = tk.Frame(
            right,
            bg="#111c32"
        )

        setup.pack(
            fill="x",
            pady=(0, 15)
        )

        details = [
            ("Engines", "70 / 15 / 15"),
            ("Test Windows", "2,629"),
            ("Window Length", "30 timesteps"),
            ("Sensor Features", "17"),
            ("Wavelet Features", "136")
        ]

        for label, value in details:

            row = tk.Frame(
                setup,
                bg="#111c32"
            )

            row.pack(
                fill="x",
                padx=18,
                pady=7
            )

            tk.Label(
                row,
                text=label,
                font=("Segoe UI", 10),
                fg="#8fa1bd",
                bg="#111c32"
            ).pack(
                side="left"
            )

            tk.Label(
                row,
                text=value,
                font=("Segoe UI", 10, "bold"),
                fg="white",
                bg="#111c32"
            ).pack(
                side="right"
            )

        # ==================================================
        # SIMULATION CONTROL
        # ==================================================

        self.create_section_title(
            right,
            "DEMONSTRATION"
        )

        self.simulation_button = tk.Button(
            right,
            text="▶  START LIVE SIMULATION",
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg="#1769aa",
            activebackground="#2186d0",
            activeforeground="white",
            relief="flat",
            padx=15,
            pady=14,
            command=self.toggle_simulation
        )

        self.simulation_button.pack(
            fill="x",
            pady=5
        )

        self.event_label = tk.Label(
            right,
            text="Waiting for simulation...",
            font=("Segoe UI", 10),
            fg="#8fa1bd",
            bg="#0b1220",
            wraplength=350
        )

        self.event_label.pack(
            pady=15
        )

        # ==================================================
        # FOOTER
        # ==================================================

        tk.Label(
            self.root,
            text=(
                "Edge AI Predictive Maintenance  •  "
                "Fault Detection + RUL Prediction"
            ),
            font=("Segoe UI", 9),
            fg="#62728d",
            bg="#0b1220"
        ).pack(
            pady=8
        )

    # ======================================================
    # UI HELPERS
    # ======================================================

    def create_section_title(
        self,
        parent,
        text
    ):

        tk.Label(
            parent,
            text=text,
            font=("Segoe UI", 11, "bold"),
            fg="#55d6ff",
            bg="#0b1220"
        ).pack(
            anchor="w",
            pady=(8, 7)
        )

    def create_model_row(
        self,
        parent,
        name,
        accuracy,
        f1,
        auc
    ):

        row = tk.Frame(
            parent,
            bg="#111c32"
        )

        row.pack(
            fill="x",
            padx=18,
            pady=10
        )

        tk.Label(
            row,
            text=name,
            width=24,
            anchor="w",
            font=("Segoe UI", 10, "bold"),
            fg="white",
            bg="#111c32"
        ).pack(
            side="left"
        )

        tk.Label(
            row,
            text=f"Accuracy {accuracy:.2f}%",
            width=18,
            font=("Segoe UI", 9),
            fg="#aab8d0",
            bg="#111c32"
        ).pack(
            side="left"
        )

        tk.Label(
            row,
            text=f"F1 {f1:.2f}%",
            width=12,
            font=("Segoe UI", 9),
            fg="#42e88b",
            bg="#111c32"
        ).pack(
            side="left"
        )

        tk.Label(
            row,
            text=f"AUC {auc:.2f}%",
            width=12,
            font=("Segoe UI", 9),
            fg="#55d6ff",
            bg="#111c32"
        ).pack(
            side="left"
        )

    def create_big_metric(
        self,
        parent,
        label,
        value
    ):

        frame = tk.Frame(
            parent,
            bg="#111c32"
        )

        frame.pack(
            fill="x",
            padx=15,
            pady=7
        )

        tk.Label(
            frame,
            text=label,
            font=("Segoe UI", 10),
            fg="#8fa1bd",
            bg="#111c32"
        ).pack(
            pady=(10, 0)
        )

        tk.Label(
            frame,
            text=value,
            font=("Segoe UI", 24, "bold"),
            fg="#55d6ff",
            bg="#111c32"
        ).pack(
            pady=(0, 10)
        )

    # ======================================================
    # LIVE SIMULATION
    # ======================================================

    def toggle_simulation(self):

        if not self.running:

            self.running = True

            self.simulation_button.config(
                text="■  STOP SIMULATION"
            )

            self.status_label.config(
                text="● LIVE SIMULATION",
                fg="#42e88b"
            )

            self.event_label.config(
                text="Monitoring sensor streams..."
            )

            self.update_simulation()

        else:

            self.running = False

            self.simulation_button.config(
                text="▶  START LIVE SIMULATION"
            )

            self.status_label.config(
                text="● SYSTEM READY",
                fg="#42e88b"
            )

            self.event_label.config(
                text="Simulation stopped."
            )

    def update_simulation(self):

        if not self.running:
            return

        self.step += 1

        # -----------------------------------------------
        # Generate realistic-looking sensor movement
        # -----------------------------------------------

        values = {}

        for sensor in self.sensor_bars:

            base = 50

            movement = (
                math.sin(self.step / 8)
                * 12
            )

            noise = random.uniform(
                -4,
                4
            )

            value = base + movement + noise

            value = max(
                5,
                min(95, value)
            )

            values[sensor] = value

        # -----------------------------------------------
        # Occasionally simulate abnormal behaviour
        # -----------------------------------------------

        fault_cycle = (
            self.step % 40
        )

        if 25 <= fault_cycle <= 32:

            values["Vibration"] = random.uniform(
                82,
                96
            )

            values["Temperature"] = random.uniform(
                75,
                90
            )

            self.machine_status.config(
                text="⚠  ANOMALY DETECTED",
                fg="#ff5c5c"
            )

            self.machine_message.config(
                text=(
                    "Abnormal sensor behaviour detected — "
                    "investigating..."
                )
            )

            self.event_label.config(
                text=(
                    "⚠ Elevated vibration and temperature "
                    "detected by monitoring layer."
                ),
                fg="#ff7b7b"
            )

        else:

            self.machine_status.config(
                text="● NORMAL",
                fg="#42e88b"
            )

            self.machine_message.config(
                text=(
                    "Machine operating within "
                    "expected range"
                )
            )

            self.event_label.config(
                text=(
                    "✓ Sensor behaviour within "
                    "expected operating range."
                ),
                fg="#8fa1bd"
            )

        # -----------------------------------------------
        # Update sensor bars
        # -----------------------------------------------

        for sensor, value in values.items():

            bar, label = self.sensor_bars[sensor]

            bar["value"] = value

            label.config(
                text=f"{value:.1f}%"
            )

        # -----------------------------------------------
        # Continue simulation
        # -----------------------------------------------

        self.root.after(
            500,
            self.update_simulation
        )

    # ======================================================
    # RUN
    # ======================================================

    def run(self):

        self.root.mainloop()


# ==========================================================
# START DASHBOARD
# ==========================================================

if __name__ == "__main__":

    dashboard = PredictiveMaintenanceDashboard()

    dashboard.run()