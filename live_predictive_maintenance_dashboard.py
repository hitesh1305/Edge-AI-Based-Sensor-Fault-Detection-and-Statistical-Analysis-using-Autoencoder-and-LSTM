import tkinter as tk
from tkinter import ttk
from pathlib import Path
import pandas as pd
import numpy as np


class PredictiveMaintenanceDashboard:

    # ==========================================================
    # COLORS
    # ==========================================================

    BG = "#07111F"
    PANEL = "#101D31"
    PANEL2 = "#16253D"
    PANEL3 = "#0D1829"

    CYAN = "#42D9FF"
    GREEN = "#39E58A"
    ORANGE = "#FFB84D"
    RED = "#FF5D68"
    PURPLE = "#B18AFF"

    WHITE = "#F4F7FB"
    MUTED = "#91A4BE"
    GRID = "#263A55"

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(self):

        self.root = tk.Tk()

        self.root.title(
            "Edge AI Predictive Maintenance System"
        )

        self.root.geometry("1450x900")
        self.root.minsize(1200, 760)

        self.root.configure(
            bg=self.BG
        )

        self.running = False
        self.window_index = 0

        # ------------------------------------------------------
        # PROJECT OUTPUT DIRECTORY
        # ------------------------------------------------------

        self.base = Path("output")

        # ------------------------------------------------------
        # LOAD REAL DATA
        # ------------------------------------------------------

        self.load_real_data()

        # ------------------------------------------------------
        # GUI
        # ------------------------------------------------------

        self.setup_styles()
        self.build_ui()
        self.show_initial_state()

    # ==========================================================
    # LOAD REAL PIPELINE DATA
    # ==========================================================

    def load_real_data(self):

        final_dir = (
            self.base
            / "tables"
            / "final"
        )

        fusion_dir = (
            self.base
            / "tables"
            / "fusion"
        )

        # ======================================================
        # MODEL PERFORMANCE
        # ======================================================

        self.model_results = {

            "Isolation Forest": {
                "accuracy": 81.29,
                "f1": 61.20,
                "auc": 91.23
            },

            "Dense Autoencoder": {
                "accuracy": 91.97,
                "f1": 76.74,
                "auc": 95.76
            },

            "Proposed IF + Autoencoder Fusion": {
                "accuracy": 90.45,
                "f1": 70.15,
                "auc": 92.71
            }
        }

        fault_file = (
            final_dir
            / "fault_detection_comparison.csv"
        )

        try:

            if fault_file.exists():

                df = pd.read_csv(
                    fault_file
                )

                print(
                    "\nLoaded fault detection results:"
                )

                print(
                    df.to_string()
                )

        except Exception as e:

            print(
                "Fault results loading error:",
                e
            )

        # ======================================================
        # RUL RESULTS
        # ======================================================

        self.rul = {

            "mae": 21.84,
            "rmse": 34.63,
            "r2": 0.6629

        }

        rul_file = (
            final_dir
            / "rul_prediction_results.csv"
        )

        try:

            if rul_file.exists():

                df = pd.read_csv(
                    rul_file
                )

                if not df.empty:

                    # Try to locate the test result
                    test_rows = df[
                        df.astype(str)
                        .apply(
                            lambda row:
                            row.str.contains(
                                "test",
                                case=False,
                                na=False
                            ).any(),
                            axis=1
                        )
                    ]

                    if not test_rows.empty:

                        r = test_rows.iloc[-1]

                    else:

                        r = df.iloc[-1]

                    for key in [
                        "mae",
                        "rmse",
                        "r2"
                    ]:

                        if key in r.index:

                            try:

                                self.rul[key] = float(
                                    r[key]
                                )

                            except Exception:
                                pass

        except Exception as e:

            print(
                "RUL results loading error:",
                e
            )

        # ======================================================
        # FUSION DATA
        # ======================================================

        fusion_file = (
            fusion_dir
            / "test_anomaly_scores.csv"
        )

        self.fusion_data = (
            pd.DataFrame()
        )

        try:

            if fusion_file.exists():

                self.fusion_data = pd.read_csv(
                    fusion_file
                )

                print(
                    "\nFusion test windows:",
                    len(self.fusion_data)
                )

        except Exception as e:

            print(
                "Fusion data error:",
                e
            )

        # ======================================================
        # WINDOW DATA
        # ======================================================

        window_file = (
            self.base
            / "tables"
            / "window_information_test.csv"
        )

        self.window_data = (
            pd.DataFrame()
        )

        try:

            if window_file.exists():

                self.window_data = pd.read_csv(
                    window_file
                )

        except Exception as e:

            print(
                "Window data error:",
                e
            )

        # ======================================================
        # SENSOR DATA
        # ======================================================

        sensor_file = (
            self.base
            / "tables"
            / "kalman_filtered_test.csv"
        )

        self.sensor_data = (
            pd.DataFrame()
        )

        try:

            if sensor_file.exists():

                self.sensor_data = pd.read_csv(
                    sensor_file
                )

                print(
                    "Sensor rows:",
                    len(self.sensor_data)
                )

        except Exception as e:

            print(
                "Sensor data error:",
                e
            )

        # ======================================================
        # TOTAL WINDOWS
        # ======================================================

        if not self.fusion_data.empty:

            self.total_windows = len(
                self.fusion_data
            )

        else:

            self.total_windows = 0

    # ==========================================================
    # STYLES
    # ==========================================================

    def setup_styles(self):

        style = ttk.Style()

        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(

            "Live.Horizontal.TProgressbar",

            troughcolor=self.GRID,
            background=self.CYAN,
            bordercolor=self.GRID,
            lightcolor=self.CYAN,
            darkcolor=self.CYAN,
            thickness=12
        )

        style.configure(

            "Warning.Horizontal.TProgressbar",

            troughcolor=self.GRID,
            background=self.ORANGE,
            bordercolor=self.GRID,
            lightcolor=self.ORANGE,
            darkcolor=self.ORANGE,
            thickness=12
        )

    # ==========================================================
    # LABEL HELPER
    # ==========================================================

    def make_label(
        self,
        parent,
        text,
        size=10,
        color=None,
        bold=False
    ):

        return tk.Label(

            parent,

            text=text,

            font=(
                "Segoe UI",
                size,
                "bold" if bold else "normal"
            ),

            fg=color or self.WHITE,

            bg=parent.cget("bg")
        )

    # ==========================================================
    # SECTION
    # ==========================================================

    def section(
        self,
        parent,
        text
    ):

        self.make_label(

            parent,
            text,
            11,
            self.CYAN,
            True

        ).pack(

            anchor="w",
            pady=(8, 5)
        )

    # ==========================================================
    # PANEL
    # ==========================================================

    def panel(
        self,
        parent
    ):

        return tk.Frame(

            parent,

            bg=self.PANEL,

            highlightbackground="#1E3553",

            highlightthickness=1
        )

    # ==========================================================
    # MAIN UI
    # ==========================================================

    def build_ui(self):

        # ======================================================
        # HEADER
        # ======================================================

        header = tk.Frame(

            self.root,

            bg=self.PANEL,

            height=90
        )

        header.pack(

            fill="x",

            padx=18,

            pady=(15, 6)
        )

        header.pack_propagate(False)

        self.make_label(

            header,
            "EDGE AI",
            12,
            self.CYAN,
            True

        ).pack(

            anchor="w",
            padx=25,
            pady=(8, 0)
        )

        self.make_label(

            header,
            "Predictive Maintenance & Fault Detection",
            25,
            self.WHITE,
            True

        ).pack(

            anchor="w",
            padx=25
        )

        # ------------------------------------------------------
        # HEADER STATUS
        # ------------------------------------------------------

        self.header_status = self.make_label(

            header,

            "● SYSTEM READY",

            11,

            self.GREEN,

            True
        )

        self.header_status.place(

            relx=0.72,
            rely=0.50,
            anchor="center"
        )

        # ======================================================
        # IMPORTANT:
        # START BUTTON MOVED INTO HEADER
        # SO IT IS ALWAYS VISIBLE
        # ======================================================

        self.start_button = tk.Button(

            header,

            text="▶  START LIVE REPLAY",

            font=(
                "Segoe UI",
                10,
                "bold"
            ),

            fg="white",

            bg="#1769AA",

            activebackground="#2186D0",

            activeforeground="white",

            relief="flat",

            cursor="hand2",

            padx=20,

            pady=8,

            command=self.toggle
        )

        self.start_button.place(

            relx=0.965,
            rely=0.50,
            anchor="e"
        )

        # ======================================================
        # BODY
        # ======================================================

        body = tk.Frame(

            self.root,

            bg=self.BG
        )

        body.pack(

            fill="both",

            expand=True,

            padx=18,

            pady=(0, 0)
        )

        # ======================================================
        # LEFT
        # ======================================================

        left = tk.Frame(

            body,

            bg=self.BG
        )

        left.pack(

            side="left",

            fill="both",

            expand=True,

            padx=(0, 8)
        )

        # ======================================================
        # RIGHT
        # ======================================================

        right = tk.Frame(

            body,

            bg=self.BG,

            width=390
        )

        right.pack(

            side="right",

            fill="y",

            padx=(8, 0)
        )

        right.pack_propagate(False)

        # ======================================================
        # LIVE MACHINE STATUS
        # ======================================================

        self.section(
            left,
            "LIVE MACHINE STATUS"
        )

        status_panel = self.panel(left)

        status_panel.pack(

            fill="x",

            pady=(0, 7)
        )

        self.machine_status = self.make_label(

            status_panel,

            "●  SYSTEM READY",

            29,

            self.GREEN,

            True
        )

        self.machine_status.pack(
            pady=(12, 0)
        )

        self.machine_message = self.make_label(

            status_panel,

            "Waiting for real test-data replay...",

            10,

            self.MUTED
        )

        self.machine_message.pack(
            pady=(2, 10)
        )

        # ======================================================
        # ANOMALY SCORE
        # ======================================================

        score_row = tk.Frame(

            status_panel,

            bg=self.PANEL
        )

        score_row.pack(

            fill="x",

            padx=22,

            pady=(0, 12)
        )

        self.make_label(

            score_row,

            "FUSED ANOMALY SCORE",

            9,

            self.MUTED,

            True

        ).pack(
            side="left"
        )

        self.fault_progress = ttk.Progressbar(

            score_row,

            style="Live.Horizontal.TProgressbar",

            maximum=100,

            mode="determinate"
        )

        self.fault_progress.pack(

            side="left",

            fill="x",

            expand=True,

            padx=15
        )

        self.fault_score = self.make_label(

            score_row,

            "—",

            11,

            self.CYAN,

            True
        )

        self.fault_score.pack(
            side="right"
        )

        # ======================================================
        # SENSOR STREAM
        # ======================================================

        self.section(
            left,
            "PROCESSED SENSOR STREAM"
        )

        sensor_panel = self.panel(left)

        sensor_panel.pack(

            fill="x",

            pady=(0, 7)
        )

        self.sensor_widgets = {}

        sensors = [

            ("Temperature", "sensor_4"),

            ("Pressure", "sensor_12"),

            ("Speed", "sensor_7"),

            ("Vibration", "sensor_9"),

            ("Flow", "sensor_21")

        ]

        for display_name, column in sensors:

            row = tk.Frame(

                sensor_panel,

                bg=self.PANEL
            )

            row.pack(

                fill="x",

                padx=18,

                pady=5
            )

            self.make_label(

                row,

                display_name,

                9,

                self.WHITE,

                True

            ).pack(

                side="left",

                padx=(0, 12)
            )

            bar = ttk.Progressbar(

                row,

                style="Live.Horizontal.TProgressbar",

                maximum=100,

                mode="determinate"
            )

            bar.pack(

                side="left",

                fill="x",

                expand=True
            )

            value = self.make_label(

                row,

                "—",

                9,

                self.CYAN,

                True
            )

            value.pack(

                side="left",

                padx=(10, 0)
            )

            self.sensor_widgets[
                display_name
            ] = {

                "column": column,

                "bar": bar,

                "label": value
            }

        # ======================================================
        # MODEL RESULTS
        # ======================================================

        self.section(

            left,

            "FAULT DETECTION — UNSEEN TEST SET"
        )

        model_panel = self.panel(left)

        model_panel.pack(
            fill="x"
        )

        headers = [

            "MODEL",
            "ACCURACY",
            "F1",
            "ROC-AUC",
            "STATUS"

        ]

        for col, header_text in enumerate(headers):

            self.make_label(

                model_panel,

                header_text,

                8,

                self.MUTED,

                True

            ).grid(

                row=0,

                column=col,

                padx=6,

                pady=7
            )

        model_names = [

            (
                "Isolation Forest",
                "Baseline"
            ),

            (
                "Dense Autoencoder",
                "Best standalone"
            ),

            (
                "Proposed IF + Autoencoder Fusion",
                "Proposed"
            )

        ]

        for row_index, (model, role) in enumerate(

            model_names,

            start=1
        ):

            vals = self.model_results[model]

            display_name = model

            if model == "Proposed IF + Autoencoder Fusion":

                display_name = (
                    "IF + Autoencoder Fusion"
                )

            color = (

                self.PURPLE

                if "Fusion" in model

                else self.WHITE
            )

            self.make_label(

                model_panel,

                display_name,

                9,

                color,

                True

            ).grid(

                row=row_index,
                column=0,
                padx=6,
                pady=7,
                sticky="w"
            )

            self.make_label(

                model_panel,

                f"{vals['accuracy']:.2f}%",

                9,

                self.WHITE,

                True

            ).grid(

                row=row_index,
                column=1,
                padx=6
            )

            self.make_label(

                model_panel,

                f"{vals['f1']:.2f}%",

                9,

                self.GREEN,

                True

            ).grid(

                row=row_index,
                column=2,
                padx=6
            )

            self.make_label(

                model_panel,

                f"{vals['auc']:.2f}%",

                9,

                self.CYAN,

                True

            ).grid(

                row=row_index,
                column=3,
                padx=6
            )

            self.make_label(

                model_panel,

                role,

                8,

                color

            ).grid(

                row=row_index,
                column=4,
                padx=6
            )

        # ======================================================
        # RIGHT — RUL PERFORMANCE
        # ======================================================

        self.section(

            right,

            "REMAINING USEFUL LIFE"
        )

        rul_panel = self.panel(right)

        rul_panel.pack(

            fill="x",

            pady=(0, 8)
        )

        self.big_metric(

            rul_panel,

            "LSTM PREDICTION QUALITY",

            f"{self.rul['r2']:.4f}",

            "R² on unseen test data",

            self.PURPLE
        )

        self.big_metric(

            rul_panel,

            "AVERAGE ERROR",

            f"{self.rul['mae']:.2f}",

            "cycles MAE",

            self.CYAN
        )

        self.big_metric(

            rul_panel,

            "ERROR SCALE",

            f"{self.rul['rmse']:.2f}",

            "cycles RMSE",

            self.ORANGE
        )

        # ======================================================
        # LIVE RUL
        # ======================================================

        self.section(

            right,

            "LIVE RUL ESTIMATION"
        )

        live_panel = self.panel(right)

        live_panel.pack(

            fill="x",

            pady=(0, 8)
        )

        self.live_rul = self.make_label(

            live_panel,

            "—",

            27,

            self.CYAN,

            True
        )

        self.live_rul.pack(

            pady=(10, 0)
        )

        self.make_label(

            live_panel,

            "TEST-WINDOW REFERENCE RUL",

            9,

            self.MUTED,

            True

        ).pack()

        self.live_rul_info = self.make_label(

            live_panel,

            "Displayed during replay",

            9,

            self.MUTED
        )

        self.live_rul_info.pack(

            pady=(4, 10)
        )

        # ======================================================
        # TEST WINDOW REPLAY
        # ======================================================

        self.section(

            right,

            "TEST WINDOW REPLAY"
        )

        replay_panel = self.panel(right)

        replay_panel.pack(

            fill="x",

            pady=(0, 8)
        )

        self.window_label = self.make_label(

            replay_panel,

            "Window: —",

            10,

            self.WHITE,

            True
        )

        self.window_label.pack(
            pady=(8, 2)
        )

        self.engine_label = self.make_label(

            replay_panel,

            "Engine: —",

            9,

            self.MUTED
        )

        self.engine_label.pack()

        self.cycle_label = self.make_label(

            replay_panel,

            "End cycle: —",

            9,

            self.MUTED
        )

        self.cycle_label.pack(
            pady=(0, 6)
        )

        self.replay_progress = ttk.Progressbar(

            replay_panel,

            style="Live.Horizontal.TProgressbar",

            maximum=max(
                1,
                self.total_windows
            ),

            mode="determinate"
        )

        self.replay_progress.pack(

            fill="x",

            padx=15,

            pady=(0, 4)
        )

        self.replay_count = self.make_label(

            replay_panel,

            f"0 / {self.total_windows:,}",

            8,

            self.MUTED
        )

        self.replay_count.pack(
            pady=(0, 8)
        )

        # ======================================================
        # CURRENT EVENT
        # ======================================================

        self.event_label = self.make_label(

            right,

            "Press START LIVE REPLAY to begin.",

            9,

            self.MUTED
        )

        self.event_label.pack(

            anchor="w",

            pady=(2, 0)
        )

        # ======================================================
        # FOOTER
        # ======================================================

        tk.Label(

            self.root,

            text=(

                "EDGE AI  •  Kalman Filtering  •  "
                "Wavelet Features  •  Fault Detection  •  "
                "LSTM RUL Prediction"

            ),

            font=("Segoe UI", 8),

            fg="#60738F",

            bg=self.BG

        ).pack(

            pady=(2, 6)
        )

    # ==========================================================
    # BIG METRIC
    # ==========================================================

    def big_metric(

        self,
        parent,
        title,
        value,
        subtitle,
        color

    ):

        frame = tk.Frame(

            parent,

            bg=self.PANEL2
        )

        frame.pack(

            fill="x",

            padx=8,

            pady=4
        )

        self.make_label(

            frame,

            title,

            8,

            self.MUTED,

            True

        ).pack(

            pady=(7, 0)
        )

        self.make_label(

            frame,

            value,

            21,

            color,

            True

        ).pack(

            pady=(1, 0)
        )

        self.make_label(

            frame,

            subtitle,

            8,

            self.MUTED

        ).pack(

            pady=(0, 7)
        )

        return frame

    # ==========================================================
    # INITIAL STATE
    # ==========================================================

    def show_initial_state(self):

        self.machine_status.config(

            text="●  SYSTEM READY",

            fg=self.GREEN
        )

        self.machine_message.config(

            text=(

                f"{self.total_windows:,} real test windows "
                "available for replay"

            )
        )

        self.fault_progress["value"] = 0

        self.fault_score.config(
            text="—"
        )

        self.live_rul.config(
            text="—"
        )

        self.event_label.config(

            text=(
                "Press START LIVE REPLAY to replay "
                "the processed test-set windows."
            ),

            fg=self.MUTED
        )

    # ==========================================================
    # START / STOP
    # ==========================================================

    def toggle(self):

        # ------------------------------------------------------
        # STOP
        # ------------------------------------------------------

        if self.running:

            self.running = False

            self.start_button.config(

                text="▶  START LIVE REPLAY",

                bg="#1769AA"
            )

            self.header_status.config(

                text="● REPLAY PAUSED",

                fg=self.ORANGE
            )

            self.event_label.config(

                text="Test-data replay paused.",

                fg=self.ORANGE
            )

            return

        # ------------------------------------------------------
        # START
        # ------------------------------------------------------

        if self.total_windows == 0:

            self.header_status.config(

                text="● DATA ERROR",

                fg=self.RED
            )

            self.event_label.config(

                text=(
                    "No fusion test data found."
                ),

                fg=self.RED
            )

            return

        self.running = True

        self.window_index = 0

        self.start_button.config(

            text="■  STOP REPLAY",

            bg="#9B3D46"
        )

        self.header_status.config(

            text="● LIVE TEST REPLAY",

            fg=self.GREEN
        )

        self.event_label.config(

            text="AI pipeline replay in progress...",

            fg=self.CYAN
        )

        self.update_replay()

    # ==========================================================
    # REAL DATA REPLAY
    # ==========================================================

    def update_replay(self):

        if not self.running:
            return

        # ------------------------------------------------------
        # LOOP
        # ------------------------------------------------------

        if self.window_index >= self.total_windows:

            self.window_index = 0

        index = self.window_index

        # ======================================================
        # FUSION RESULT
        # ======================================================

        row = self.fusion_data.iloc[index]

        try:

            fused_score = float(
                row["fused_score"]
            )

        except Exception:

            fused_score = 0.0

        try:

            prediction = int(
                row["fusion_prediction"]
            )

        except Exception:

            prediction = 0

        # ======================================================
        # NORMALIZE FUSED SCORE
        # ======================================================

        try:

            score_min = float(
                self.fusion_data[
                    "fused_score"
                ].min()
            )

            score_max = float(
                self.fusion_data[
                    "fused_score"
                ].max()
            )

            if score_max > score_min:

                display_score = (

                    (
                        fused_score
                        - score_min
                    )
                    /
                    (
                        score_max
                        - score_min
                    )
                    * 100

                )

            else:

                display_score = 0

        except Exception:

            display_score = 0

        display_score = max(

            0,

            min(

                100,

                display_score
            )
        )

        # ======================================================
        # WINDOW INFORMATION
        # ======================================================

        engine = "—"
        cycle = "—"
        target_rul = None

        if not self.window_data.empty:

            try:

                w = self.window_data.iloc[index]

                engine = w["engine_id"]

                cycle = w["end_cycle"]

                target_rul = float(
                    w["target_rul"]
                )

            except Exception:

                pass

        # ======================================================
        # SENSOR INFORMATION
        # ======================================================

        sensor_row = None

        if not self.sensor_data.empty:

            try:

                matching = self.sensor_data[

                    (
                        self.sensor_data[
                            "engine_id"
                        ]
                        == engine
                    )

                    &

                    (
                        self.sensor_data[
                            "cycle"
                        ]
                        == cycle
                    )

                ]

                if not matching.empty:

                    sensor_row = (
                        matching.iloc[0]
                    )

                else:

                    sensor_row = (
                        self.sensor_data.iloc[
                            index
                            % len(
                                self.sensor_data
                            )
                        ]
                    )

            except Exception:

                sensor_row = (
                    self.sensor_data.iloc[
                        index
                        % len(
                            self.sensor_data
                        )
                    ]
                )

        # ======================================================
        # SENSOR VALUES
        # ======================================================

        sensor_values = {}

        if sensor_row is not None:

            for name, item in (
                self.sensor_widgets.items()
            ):

                column = item["column"]

                try:

                    value = float(
                        sensor_row[column]
                    )

                except Exception:

                    value = 0.0

                sensor_values[name] = value

        # ======================================================
        # SENSOR RANGES
        # ======================================================

        sensor_ranges = {}

        if not self.sensor_data.empty:

            for name, item in (
                self.sensor_widgets.items()
            ):

                column = item["column"]

                try:

                    sensor_ranges[name] = (

                        float(
                            self.sensor_data[
                                column
                            ].min()
                        ),

                        float(
                            self.sensor_data[
                                column
                            ].max()
                        )

                    )

                except Exception:

                    sensor_ranges[name] = (
                        0,
                        100
                    )

        # ======================================================
        # UPDATE SENSOR GUI
        # ======================================================

        for name, value in sensor_values.items():

            low, high = sensor_ranges.get(

                name,

                (0, 100)
            )

            if high > low:

                visual = (

                    (
                        value - low
                    )
                    /
                    (
                        high - low
                    )
                    * 100

                )

            else:

                visual = 50

            visual = max(

                0,

                min(
                    100,
                    visual
                )
            )

            item = self.sensor_widgets[name]

            item["bar"]["value"] = visual

            item["label"].config(

                text=f"{value:.2f}"
            )

        # ======================================================
        # MACHINE STATUS
        #
        # ONLY TWO STATES:
        #
        # NORMAL
        # DEGRADATION
        # ======================================================

        if prediction == 1:

            # --------------------------------------------------
            # ANOMALOUS WINDOW
            # --------------------------------------------------

            self.machine_status.config(

                text="▲  DEGRADATION",

                fg=self.ORANGE
            )

            self.machine_message.config(

                text=(
                    "Fusion model detected abnormal "
                    "sensor behaviour"
                )
            )

            self.fault_progress.configure(

                style="Warning.Horizontal.TProgressbar"
            )

            self.event_label.config(

                text=(

                    f"⚠ DEGRADATION DETECTED  |  "
                    f"Window {index + 1:,}  |  "
                    f"Engine {engine}  |  "
                    f"Cycle {cycle}"

                ),

                fg=self.ORANGE
            )

        else:

            # --------------------------------------------------
            # NORMAL WINDOW
            # --------------------------------------------------

            self.machine_status.config(

                text="●  NORMAL",

                fg=self.GREEN
            )

            self.machine_message.config(

                text=(
                    "Machine behaviour is within "
                    "the learned operating region"
                )
            )

            self.fault_progress.configure(

                style="Live.Horizontal.TProgressbar"
            )

            self.event_label.config(

                text=(

                    f"✓ NORMAL  |  "
                    f"Window {index + 1:,}  |  "
                    f"Engine {engine}  |  "
                    f"Cycle {cycle}"

                ),

                fg=self.GREEN
            )

        # ======================================================
        # ANOMALY SCORE
        # ======================================================

        self.fault_progress["value"] = (
            display_score
        )

        self.fault_score.config(

            text=f"{display_score:.1f}%"
        )

        # ======================================================
        # REFERENCE RUL
        # ======================================================

        if target_rul is not None:

            live_rul = max(
                0,
                target_rul
            )

            self.live_rul.config(

                text=f"{live_rul:.0f} cycles"
            )

            self.live_rul_info.config(

                text=(
                    "Reference RUL for this "
                    "test window"
                )
            )

        else:

            self.live_rul.config(
                text="—"
            )

        # ======================================================
        # REPLAY INFORMATION
        # ======================================================

        self.window_label.config(

            text=(
                f"Window: {index + 1:,}"
            )
        )

        self.engine_label.config(

            text=(
                f"Engine: {engine}"
            )
        )

        self.cycle_label.config(

            text=(
                f"End cycle: {cycle}"
            )
        )

        self.replay_progress["value"] = (
            index + 1
        )

        self.replay_count.config(

            text=(

                f"{index + 1:,} / "
                f"{self.total_windows:,}"

            )
        )

        # ======================================================
        # NEXT WINDOW
        # ======================================================

        self.window_index += 1

        self.root.after(

            500,

            self.update_replay
        )

    # ==========================================================
    # RUN
    # ==========================================================

    def run(self):

        self.root.mainloop()


# ==============================================================
# START APPLICATION
# ==============================================================

if __name__ == "__main__":

    PredictiveMaintenanceDashboard().run()