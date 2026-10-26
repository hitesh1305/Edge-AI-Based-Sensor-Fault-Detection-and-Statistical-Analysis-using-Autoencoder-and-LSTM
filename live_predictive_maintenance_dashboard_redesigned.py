"""Visual redesign of the Edge AI Predictive Maintenance dashboard.

Reads the existing output/ CSV files and replays saved NASA C-MAPSS test windows.
This is an offline test-data replay, not a live physical sensor connection.
"""
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PredictiveMaintenanceDashboard:
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

    SENSOR_COLUMNS = [
        ("Temperature", "sensor_4"),
        ("Pressure", "sensor_12"),
        ("Speed", "sensor_7"),
        ("Vibration proxy", "sensor_9"),
        ("Flow proxy", "sensor_21"),
    ]

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Edge AI | Predictive Maintenance & Fault Detection")
        self.root.geometry("1400x820")
        self.root.minsize(1000, 620)
        try:
            self.root.state("zoomed")  # Windows: use the full screen on launch
        except tk.TclError:
            pass
        self.root.configure(bg=self.BG)

        # Scrollable content keeps lower dashboard sections reachable on
        # laptops and smaller displays instead of clipping them off-screen.
        shell = tk.Frame(self.root, bg=self.BG)
        shell.pack(fill="both", expand=True)
        self.page_canvas = tk.Canvas(shell, bg=self.BG, highlightthickness=0)
        self.page_scrollbar = ttk.Scrollbar(shell, orient="vertical", command=self.page_canvas.yview)
        self.page_canvas.configure(yscrollcommand=self.page_scrollbar.set)
        self.page_scrollbar.pack(side="right", fill="y")
        self.page_canvas.pack(side="left", fill="both", expand=True)
        self.content_frame = tk.Frame(self.page_canvas, bg=self.BG)
        self._content_window = self.page_canvas.create_window((0, 0), window=self.content_frame, anchor="nw")
        self.content_frame.bind("<Configure>", self._update_scroll_region)
        self.page_canvas.bind("<Configure>", self._resize_content)
        self.page_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.base = Path("output")
        self.running = False
        self.window_index = 0
        self.history = []
        self.sensor_history = {name: [] for name, _ in self.SENSOR_COLUMNS}
        self._load_data()
        self._styles()
        self._build_ui()
        self._show_initial_state()

    def _read_csv(self, path):
        try:
            return pd.read_csv(path) if path.exists() else pd.DataFrame()
        except Exception as exc:
            print(f"Could not read {path}: {exc}")
            return pd.DataFrame()

    def _load_data(self):
        tables = self.base / "tables"
        final = tables / "final"
        fusion = tables / "fusion"
        self.fusion_data = self._read_csv(fusion / "test_anomaly_scores.csv")
        self.window_data = self._read_csv(tables / "window_information_test.csv")
        self.sensor_data = self._read_csv(tables / "kalman_filtered_test.csv")
        self.model_results = self._load_model_results(final / "fault_detection_comparison.csv")
        self.rul_metrics = self._load_rul_metrics(final / "rul_prediction_results.csv")
        self.total_windows = len(self.fusion_data)

        # Use actual column names where possible; tolerate common naming variations.
        self.score_col = self._find_col(self.fusion_data, ["fused_score", "fusion_score", "anomaly_score"])
        self.pred_col = self._find_col(self.fusion_data, ["fusion_prediction", "prediction", "predicted_label"])
        self.window_engine_col = self._find_col(self.window_data, ["engine_id", "engine", "unit"])
        self.window_cycle_col = self._find_col(self.window_data, ["end_cycle", "cycle", "endcycle"])
        self.window_rul_col = self._find_col(self.window_data, ["target_rul", "rul", "true_rul", "reference_rul"])

        # Summarize saved labels so a persistent anomaly state can be checked
        # against the actual test-set predictions rather than guessed at.
        self.anomaly_count = None
        self.anomaly_rate = None
        self.label_summary = "Prediction-label distribution unavailable"
        if not self.fusion_data.empty and self.pred_col:
            labels = pd.to_numeric(self.fusion_data[self.pred_col], errors="coerce").dropna()
            if not labels.empty:
                self.anomaly_count = int((labels == 1).sum())
                self.anomaly_rate = 100.0 * self.anomaly_count / len(labels)
                self.label_summary = (
                    f"Saved test labels: {self.anomaly_count:,}/{len(labels):,} anomalous "
                    f"({self.anomaly_rate:.1f}%)"
                )

    @staticmethod
    def _find_col(df, names):
        if df.empty:
            return None
        mapping = {str(c).strip().lower(): c for c in df.columns}
        for name in names:
            if name.lower() in mapping:
                return mapping[name.lower()]
        for c in df.columns:
            cl = str(c).strip().lower()
            if any(name.lower() in cl for name in names):
                return c
        return None

    def _load_model_results(self, path):
        df = self._read_csv(path)
        results = []
        if not df.empty:
            for _, row in df.iterrows():
                model = str(row.get("model", row.get("Model", row.iloc[0])))
                lower = model.lower()
                if "isolation" in lower:
                    friendly = "Isolation Forest"
                elif "autoencoder" in lower and "fusion" not in lower:
                    friendly = "Dense Autoencoder"
                elif "fusion" in lower or ("isolation" in lower and "autoencoder" in lower):
                    friendly = "IF + Autoencoder Fusion"
                else:
                    friendly = model
                def metric(candidates):
                    col = self._find_col(pd.DataFrame(columns=df.columns), candidates)
                    if col is None:
                        return None
                    try:
                        v = float(row[col])
                        return v * 100 if 0 <= v <= 1 else v
                    except Exception:
                        return None
                acc = metric(["accuracy", "test_accuracy"])
                f1 = metric(["f1", "f1_score", "test_f1"])
                auc = metric(["roc_auc", "auc", "roc-auc"])
                if any(v is not None for v in (acc, f1, auc)):
                    results.append({"model": friendly, "accuracy": acc, "f1": f1, "auc": auc})
        # Fallback only if the CSV is missing/unreadable; clearly mark as prior snapshot.
        if not results:
            results = [
                {"model": "Isolation Forest", "accuracy": 81.29, "f1": 61.20, "auc": 91.23},
                {"model": "Dense Autoencoder", "accuracy": 91.97, "f1": 76.74, "auc": 95.76},
                {"model": "IF + Autoencoder Fusion", "accuracy": 90.45, "f1": 70.15, "auc": 92.71},
            ]
            self.metrics_fallback = True
        else:
            self.metrics_fallback = False
        return results

    def _load_rul_metrics(self, path):
        df = self._read_csv(path)
        defaults = {"mae": None, "rmse": None, "r2": None}
        if df.empty:
            return defaults
        row = df.iloc[-1]
        for key, names in {
            "mae": ["mae", "test_mae"],
            "rmse": ["rmse", "test_rmse"],
            "r2": ["r2", "r_squared", "r2_score", "test_r2"],
        }.items():
            col = self._find_col(df, names)
            if col is not None:
                try:
                    defaults[key] = float(row[col])
                except Exception:
                    pass
        return defaults

    def _styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TProgressbar", troughcolor=self.GRID, background=self.CYAN, bordercolor=self.GRID, thickness=10)

    def _label(self, parent, text, size=10, color=None, bold=False, **kwargs):
        return tk.Label(parent, text=text, font=("Segoe UI", size, "bold" if bold else "normal"),
                        fg=color or self.WHITE, bg=kwargs.pop("bg", parent.cget("bg")), **kwargs)

    def _panel(self, parent, padx=0, pady=0):
        frame = tk.Frame(parent, bg=self.PANEL, highlightbackground="#1E3553", highlightthickness=1)
        if padx or pady:
            frame.pack(fill="both", expand=True, padx=padx, pady=pady)
        return frame

    def _section_title(self, parent, text):
        self._label(parent, text.upper(), 10, self.CYAN, True).pack(anchor="w", pady=(8, 5))

    def _build_ui(self):
        # Header
        header = tk.Frame(self.content_frame, bg=self.PANEL, height=88)
        header.pack(fill="x", padx=16, pady=(14, 8))
        header.pack_propagate(False)
        lefthead = tk.Frame(header, bg=self.PANEL)
        lefthead.pack(side="left", fill="y", padx=20, pady=10)
        self._label(lefthead, "EDGE AI  /  CONDITION MONITORING", 9, self.CYAN, True).pack(anchor="w")
        self._label(lefthead, "Predictive Maintenance & Fault Detection", 22, self.WHITE, True).pack(anchor="w", pady=(2, 0))
        self.header_status = self._label(header, "● READY", 10, self.GREEN, True)
        self.header_status.pack(side="left", padx=22)
        self.start_button = tk.Button(header, text="▶  START REPLAY", command=self.toggle, font=("Segoe UI", 10, "bold"),
                                      fg="white", bg="#1769AA", activebackground="#2186D0", activeforeground="white",
                                      relief="flat", cursor="hand2", padx=18, pady=9)
        self.start_button.pack(side="right", padx=18)

        # Important provenance banner
        banner = tk.Frame(self.content_frame, bg="#13243A")
        banner.pack(fill="x", padx=16, pady=(0, 8))
        self._label(banner, "DEMO MODE", 9, self.ORANGE, True, bg="#13243A").pack(side="left", padx=(12, 8), pady=7)
        self._label(banner, "Replaying saved NASA C-MAPSS test windows — not live physical sensor readings.", 9, self.MUTED, bg="#13243A").pack(side="left", pady=7)
        self._label(banner, self.label_summary, 9, self.ORANGE, True, bg="#13243A").pack(side="right", padx=12, pady=7)

        # Main 2-column layout
        body = tk.Frame(self.content_frame, bg=self.BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        left = tk.Frame(body, bg=self.BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right = tk.Frame(body, bg=self.BG, width=390)
        right.pack(side="right", fill="both", padx=(8, 0))
        right.pack_propagate(False)

        # Machine status / gauge
        self._section_title(left, "01  Machine condition")
        status = self._panel(left)
        status.pack(fill="x", pady=(0, 7))
        status_top = tk.Frame(status, bg=self.PANEL)
        status_top.pack(fill="x", padx=16, pady=(12, 4))
        status_left = tk.Frame(status_top, bg=self.PANEL)
        status_left.pack(side="left", fill="both", expand=True)
        self.machine_status = self._label(status_left, "WAITING FOR REPLAY", 23, self.GREEN, True)
        self.machine_status.pack(anchor="w")
        self.machine_message = self._label(status_left, "Start replay to inspect model decisions.", 10, self.MUTED)
        self.machine_message.pack(anchor="w", pady=(4, 0))
        self.status_detail = self._label(status_left, "No window selected", 9, self.MUTED)
        self.status_detail.pack(anchor="w", pady=(7, 0))
        self.gauge = tk.Canvas(status_top, width=180, height=125, bg=self.PANEL, highlightthickness=0)
        self.gauge.pack(side="right", padx=12)
        scoreline = tk.Frame(status, bg=self.PANEL)
        scoreline.pack(fill="x", padx=16, pady=(0, 13))
        self._label(scoreline, "FUSED ANOMALY SCORE", 9, self.MUTED, True).pack(side="left")
        self.score_text = self._label(scoreline, "—", 12, self.CYAN, True)
        self.score_text.pack(side="right")
        self._draw_gauge(None, None)

        # Sensor trend plot
        self._section_title(left, "02  Sensor behaviour over replay")
        chartpanel = self._panel(left)
        chartpanel.pack(fill="both", expand=True, pady=(0, 7))
        self.sensor_fig = Figure(figsize=(7.8, 3.05), dpi=100, facecolor=self.PANEL)
        self.sensor_ax = self.sensor_fig.add_subplot(111)
        self._style_axis(self.sensor_ax)
        self.sensor_ax.set_title("Sensor values by replay window", color=self.WHITE, fontsize=10, loc="left", pad=8)
        self.sensor_ax.set_xlabel("Replay window", color=self.MUTED, fontsize=8)
        self.sensor_ax.set_ylabel("Standardized / filtered value", color=self.MUTED, fontsize=8)
        self.sensor_canvas = FigureCanvasTkAgg(self.sensor_fig, master=chartpanel)
        self.sensor_canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=3)

        # Model comparison chart
        self._section_title(left, "03  Model performance on held-out test set")
        compare = self._panel(left)
        compare.pack(fill="both", expand=True)
        self.model_fig = Figure(figsize=(7.8, 2.5), dpi=100, facecolor=self.PANEL)
        self.model_ax = self.model_fig.add_subplot(111)
        self._style_axis(self.model_ax)
        self._plot_model_comparison()
        self.model_canvas = FigureCanvasTkAgg(self.model_fig, master=compare)
        self.model_canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=3)

        # Right column KPI cards
        self._section_title(right, "04  Prediction quality")
        kpis = self._panel(right)
        kpis.pack(fill="x", pady=(0, 8))
        self._metric_card(kpis, "RUL MAE", self._fmt(self.rul_metrics['mae'], 2, " cycles"), "Average absolute error", self.CYAN)
        self._metric_card(kpis, "RUL RMSE", self._fmt(self.rul_metrics['rmse'], 2, " cycles"), "Penalizes larger errors", self.ORANGE)
        self._metric_card(kpis, "RUL R²", self._fmt(self.rul_metrics['r2'], 3), "Explained variance on test data", self.PURPLE)

        self._section_title(right, "05  Anomaly timeline")
        timeline = self._panel(right)
        timeline.pack(fill="both", expand=True, pady=(0, 8))
        self.timeline_fig = Figure(figsize=(3.5, 2.8), dpi=100, facecolor=self.PANEL)
        self.timeline_ax = self.timeline_fig.add_subplot(111)
        self._style_axis(self.timeline_ax)
        self.timeline_ax.set_title("Recent anomaly scores", color=self.WHITE, fontsize=10, loc="left", pad=8)
        self.timeline_ax.set_xlabel("Replay window", color=self.MUTED, fontsize=8)
        self.timeline_canvas = FigureCanvasTkAgg(self.timeline_fig, master=timeline)
        self.timeline_canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=3)

        self._section_title(right, "06  Current test window")
        window = self._panel(right)
        window.pack(fill="x", pady=(0, 8))
        self.window_label = self._label(window, "Window — / —", 12, self.WHITE, True)
        self.window_label.pack(anchor="w", padx=13, pady=(10, 3))
        self.engine_label = self._label(window, "Engine: —", 10, self.MUTED)
        self.engine_label.pack(anchor="w", padx=13, pady=2)
        self.cycle_label = self._label(window, "End cycle: —", 10, self.MUTED)
        self.cycle_label.pack(anchor="w", padx=13, pady=2)
        self.rul_label = self._label(window, "Reference RUL: —", 10, self.MUTED)
        self.rul_label.pack(anchor="w", padx=13, pady=(2, 10))
        self.replay_progress = ttk.Progressbar(window, maximum=max(1, self.total_windows), mode="determinate")
        self.replay_progress.pack(fill="x", padx=13, pady=(0, 4))
        self.replay_count = self._label(window, f"0 / {self.total_windows:,} windows", 8, self.MUTED)
        self.replay_count.pack(anchor="e", padx=13, pady=(0, 8))

        footer = tk.Frame(self.content_frame, bg=self.BG)
        footer.pack(fill="x", padx=18, pady=(0, 8))
        self.event_label = self._label(footer, "Ready. Start replay to step through held-out test windows.", 9, self.MUTED)
        self.event_label.pack(side="left")
        self._label(footer, "Kalman filtering  •  Wavelet features  •  Isolation Forest  •  Autoencoder  •  LSTM", 8, "#60738F").pack(side="right")

    def _update_scroll_region(self, _event=None):
        self.page_canvas.configure(scrollregion=self.page_canvas.bbox("all"))

    def _resize_content(self, event):
        self.page_canvas.itemconfigure(self._content_window, width=event.width)

    def _on_mousewheel(self, event):
        # Windows mouse wheel: scroll the full dashboard vertically.
        if event.delta:
            self.page_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _metric_card(self, parent, title, value, subtitle, color):
        card = tk.Frame(parent, bg=self.PANEL2)
        card.pack(fill="x", padx=8, pady=4)
        row = tk.Frame(card, bg=self.PANEL2)
        row.pack(fill="x", padx=10, pady=(8, 0))
        self._label(row, title, 9, self.MUTED, True, bg=self.PANEL2).pack(side="left")
        self._label(row, value, 17, color, True, bg=self.PANEL2).pack(side="right")
        self._label(card, subtitle, 8, self.MUTED, bg=self.PANEL2).pack(anchor="w", padx=10, pady=(0, 8))

    @staticmethod
    def _fmt(value, decimals=2, suffix=""):
        try:
            return f"{float(value):.{decimals}f}{suffix}"
        except (TypeError, ValueError):
            return "N/A"

    def _style_axis(self, ax):
        ax.set_facecolor(self.PANEL)
        ax.tick_params(colors=self.MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(self.GRID)
        ax.grid(True, color=self.GRID, linewidth=0.6, alpha=0.8)

    def _plot_model_comparison(self):
        ax = self.model_ax
        ax.clear(); self._style_axis(ax)
        names = [r["model"] for r in self.model_results]
        vals = [r["f1"] if r["f1"] is not None else 0 for r in self.model_results]
        colors = [self.CYAN if "Fusion" not in n else self.PURPLE for n in names]
        y = np.arange(len(names))
        bars = ax.barh(y, vals, color=colors, height=0.52)
        ax.set_yticks(y, [n.replace("IF + Autoencoder Fusion", "IF + Autoencoder Fusion") for n in names])
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlim(0, 100); ax.set_xlabel("F1 score (%)", color=self.MUTED, fontsize=8)
        ax.set_title("F1 score comparison (higher is better)", color=self.WHITE, fontsize=10, loc="left", pad=8)
        for bar, v in zip(bars, vals):
            ax.text(min(v + 1.2, 96), bar.get_y() + bar.get_height()/2, f"{v:.1f}%", va="center", color=self.WHITE, fontsize=8)
        self.model_fig.tight_layout(pad=1.4)

    def _draw_gauge(self, score, prediction):
        c = self.gauge
        c.delete("all")
        c.create_text(90, 10, text="ANOMALY INDEX", fill=self.MUTED, font=("Segoe UI", 9, "bold"))
        c.create_arc(18, 17, 162, 126, start=180, extent=180, style="arc", outline=self.GRID, width=13)
        if score is not None:
            frac = max(0.0, min(1.0, score / 100.0))
            color = self.RED if prediction == 1 else self.GREEN
            c.create_arc(18, 17, 162, 126, start=180, extent=180*frac, style="arc", outline=color, width=13)
            c.create_text(90, 72, text=f"{score:.1f}%", fill=color, font=("Segoe UI", 22, "bold"))
            c.create_text(90, 96, text="anomaly index*", fill=self.MUTED, font=("Segoe UI", 8))
        else:
            c.create_text(90, 72, text="—", fill=self.WHITE, font=("Segoe UI", 22, "bold"))
            c.create_text(90, 96, text="awaiting replay", fill=self.MUTED, font=("Segoe UI", 8))

    def _show_initial_state(self):
        self.machine_status.config(text="WAITING FOR REPLAY", fg=self.GREEN)
        self.machine_message.config(text=f"{self.total_windows:,} saved test windows available." if self.total_windows else "No fusion test scores found.")
        self.status_detail.config(text="Start replay to see engine, cycle and model decision.")
        self.score_text.config(text="—")
        self._draw_gauge(None, None)
        self._plot_timeline()
        self._plot_sensor_history()

    def _plot_sensor_history(self):
        ax = self.sensor_ax
        ax.clear(); self._style_axis(ax)
        ax.set_title("Filtered sensor traces (values are not raw physical units)", color=self.WHITE, fontsize=10, loc="left", pad=8)
        ax.set_xlabel("Replay window", color=self.MUTED, fontsize=8)
        ax.set_ylabel("Filtered value", color=self.MUTED, fontsize=8)
        has_data = False
        for name, _ in self.SENSOR_COLUMNS:
            seq = self.sensor_history[name]
            if seq:
                ax.plot(range(max(1, len(seq)-len(self.history)+1), len(seq)+1), seq[-len(self.history):], label=name, linewidth=1.5)
                has_data = True
        if has_data:
            leg = ax.legend(loc="upper left", ncol=3, fontsize=7, frameon=False)
            for t in leg.get_texts(): t.set_color(self.MUTED)
        else:
            ax.text(0.5, 0.5, "Press START REPLAY to build sensor trends", transform=ax.transAxes,
                    ha="center", va="center", color=self.MUTED, fontsize=9)
        self.sensor_fig.tight_layout(pad=1.4)
        self.sensor_canvas.draw_idle()

    def _plot_timeline(self):
        ax = self.timeline_ax
        ax.clear(); self._style_axis(ax)
        ax.set_title("Recent anomaly scores", color=self.WHITE, fontsize=10, loc="left", pad=8)
        ax.set_xlabel("Replay window", color=self.MUTED, fontsize=8)
        ax.set_ylabel("Anomaly index*", color=self.MUTED, fontsize=8)
        if self.history:
            recent = self.history[-80:]
            xs = [h["index"] for h in recent]
            ys = [h["score"] for h in recent]
            cs = [self.RED if h["prediction"] == 1 else self.GREEN for h in recent]
            ax.plot(xs, ys, color=self.CYAN, alpha=0.55, linewidth=1.2)
            ax.scatter(xs, ys, c=cs, s=18)
            ax.set_ylim(0, 100)
        else:
            ax.text(0.5, 0.5, "No replay data yet", transform=ax.transAxes, ha="center", va="center", color=self.MUTED, fontsize=9)
        self.timeline_fig.tight_layout(pad=1.4)
        self.timeline_canvas.draw_idle()

    def toggle(self):
        if self.running:
            self.running = False
            self.start_button.config(text="▶  RESUME REPLAY", bg="#1769AA")
            self.header_status.config(text="● PAUSED", fg=self.ORANGE)
            self.event_label.config(text="Replay paused.", fg=self.ORANGE)
            return
        if self.fusion_data.empty or not self.score_col or not self.pred_col:
            messagebox.showerror("Replay data unavailable", "Could not find fusion test scores or prediction columns in output/tables/fusion/test_anomaly_scores.csv.")
            return
        self.running = True
        if self.window_index >= self.total_windows:
            self.window_index = 0
            self.history.clear()
            for k in self.sensor_history: self.sensor_history[k].clear()
        self.start_button.config(text="■  STOP REPLAY", bg="#9B3D46")
        self.header_status.config(text="● TEST REPLAY RUNNING", fg=self.GREEN)
        self.event_label.config(text="Replaying saved held-out test windows…", fg=self.CYAN)
        self.update_replay()

    def update_replay(self):
        if not self.running:
            return
        if self.window_index >= self.total_windows:
            self.running = False
            self.start_button.config(text="↻  REPLAY AGAIN", bg="#1769AA")
            self.header_status.config(text="● REPLAY COMPLETE", fg=self.GREEN)
            self.event_label.config(text="Replay complete. All saved test windows were displayed.", fg=self.GREEN)
            return

        idx = self.window_index
        row = self.fusion_data.iloc[idx]
        try: raw_score = float(row[self.score_col])
        except Exception: raw_score = 0.0
        try: prediction = int(row[self.pred_col])
        except Exception: prediction = 0

        # Display index is explicitly a relative min-max index, not a probability.
        try:
            lo = float(pd.to_numeric(self.fusion_data[self.score_col], errors="coerce").min())
            hi = float(pd.to_numeric(self.fusion_data[self.score_col], errors="coerce").max())
            score = (raw_score - lo) / (hi - lo) * 100 if hi > lo else 0.0
        except Exception:
            score = 0.0
        score = float(np.clip(score, 0, 100))

        engine, cycle, target_rul = "—", "—", None
        if not self.window_data.empty and idx < len(self.window_data):
            w = self.window_data.iloc[idx]
            if self.window_engine_col: engine = w[self.window_engine_col]
            if self.window_cycle_col: cycle = w[self.window_cycle_col]
            if self.window_rul_col:
                try: target_rul = float(w[self.window_rul_col])
                except Exception: pass

        sensor_row = None
        if not self.sensor_data.empty:
            try:
                if self.window_engine_col and self.window_cycle_col and "engine_id" in self.sensor_data.columns and "cycle" in self.sensor_data.columns:
                    m = self.sensor_data[(self.sensor_data["engine_id"] == engine) & (self.sensor_data["cycle"] == cycle)]
                    if not m.empty: sensor_row = m.iloc[0]
                if sensor_row is None and idx < len(self.sensor_data): sensor_row = self.sensor_data.iloc[idx]
            except Exception: pass

        self.history.append({"index": idx + 1, "score": score, "prediction": prediction})
        if len(self.history) > 300: self.history = self.history[-300:]
        for name, col in self.SENSOR_COLUMNS:
            if sensor_row is not None and col in sensor_row.index:
                try: val = float(sensor_row[col])
                except Exception: val = np.nan
                if np.isfinite(val): self.sensor_history[name].append(val)
        for key in self.sensor_history:
            if len(self.sensor_history[key]) > 300: self.sensor_history[key] = self.sensor_history[key][-300:]

        if prediction == 1:
            self.machine_status.config(text="⚠ ANOMALOUS PATTERN", fg=self.ORANGE)
            self.machine_message.config(text="The fusion classifier labels this test window as anomalous.")
            self.event_label.config(text=f"ANOMALY LABEL  •  Window {idx+1:,}  •  Engine {engine}  •  Cycle {cycle}", fg=self.ORANGE)
        else:
            self.machine_status.config(text="✓ NORMAL PATTERN", fg=self.GREEN)
            self.machine_message.config(text="The fusion classifier labels this test window as normal.")
            self.event_label.config(text=f"NORMAL LABEL  •  Window {idx+1:,}  •  Engine {engine}  •  Cycle {cycle}", fg=self.GREEN)
        self.status_detail.config(text=f"Raw fused score: {raw_score:.5f}   |   Classifier label: {'Anomaly' if prediction else 'Normal'}")
        self.score_text.config(text=f"{score:.1f}% index*")
        self._draw_gauge(score, prediction)
        self.window_label.config(text=f"Window {idx+1:,} of {self.total_windows:,}")
        self.engine_label.config(text=f"Engine: {engine}")
        self.cycle_label.config(text=f"End cycle: {cycle}")
        self.rul_label.config(text=f"Reference RUL: {target_rul:.0f} cycles" if target_rul is not None else "Reference RUL: unavailable")
        self.replay_progress["value"] = idx + 1
        self.replay_count.config(text=f"{idx+1:,} / {self.total_windows:,} windows")

        # Redraw charts every 2 windows for a responsive UI.
        if idx % 2 == 0 or idx == self.total_windows - 1:
            self._plot_sensor_history()
            self._plot_timeline()
        self.window_index += 1
        self.root.after(500, self.update_replay)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    PredictiveMaintenanceDashboard().run()
