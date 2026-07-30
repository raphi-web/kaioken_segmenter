"""Minimal Tkinter GUI for the standalone predictor.

Flow: pick a model.onnx (auto-detected next to the executable) and a GeoTIFF,
map each model input channel to a band of the image (pre-filled identity when
the counts match), choose an output path, and Run. Prediction happens on a
worker thread; progress and result are marshalled back through a queue.
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import core

_TIF_TYPES = [("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")]


class PredictorGUI:
    def __init__(self, root):
        self.root = root
        root.title("U-Net Predictor")
        self.model = None          # loaded model dict, or None
        self.info = None           # input band info, or None
        self.combos = []           # one ttk.Combobox per model channel
        self.queue = queue.Queue()
        self.model_path = tk.StringVar()
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self._build()
        default = core.default_model_path()
        if os.path.exists(default):
            self._set_model(default)

    # ---------- layout ----------

    def _build(self):
        frm = ttk.Frame(self.root, padding=12)
        frm.grid(sticky="nsew")
        frm.columnconfigure(1, weight=1)

        self._path_row(frm, 0, "Model (.onnx)", self.model_path, self._pick_model)
        self._path_row(frm, 1, "Input GeoTIFF", self.input_path, self._pick_input)
        self._path_row(frm, 2, "Output", self.output_path, self._pick_output)

        self.bands_frame = ttk.LabelFrame(frm, text="Band mapping", padding=8)
        self.bands_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        self.bands_frame.columnconfigure(1, weight=1)
        self.bands_hint = ttk.Label(self.bands_frame,
                                    text="Select a model and an input image.")
        self.bands_hint.grid(row=0, column=0, sticky="w")

        self.run_btn = ttk.Button(frm, text="Run", command=self._run, state="disabled")
        self.run_btn.grid(row=4, column=0, sticky="w", pady=(6, 2))
        self.progress = ttk.Progressbar(frm, mode="determinate")
        self.progress.grid(row=4, column=1, columnspan=2, sticky="ew", padx=(8, 0))
        self.status = ttk.Label(frm, text="", foreground="#555")
        self.status.grid(row=5, column=0, columnspan=3, sticky="w")

    def _path_row(self, frm, row, label, var, command):
        ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(frm, textvariable=var, width=52, state="readonly").grid(
            row=row, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(frm, text="Browse…", command=command).grid(row=row, column=2, pady=3)

    # ---------- pickers ----------

    def _pick_model(self):
        path = filedialog.askopenfilename(title="Select model.onnx",
                                          filetypes=[("ONNX model", "*.onnx")])
        if path:
            self._set_model(path)

    def _pick_input(self):
        path = filedialog.askopenfilename(title="Select input GeoTIFF",
                                          filetypes=_TIF_TYPES)
        if path:
            self._set_input(path)

    def _pick_output(self):
        path = filedialog.asksaveasfilename(title="Save prediction as",
                                            defaultextension=".tif",
                                            filetypes=_TIF_TYPES)
        if path:
            self.output_path.set(path)
            self._update_run_state()

    # ---------- state ----------

    def _set_model(self, path):
        try:
            self.model = core.load_model(path)
        except Exception as e:
            messagebox.showerror("Could not load model", str(e))
            return
        self.model_path.set(path)
        self._rebuild_bands()
        self._update_run_state()

    def _set_input(self, path):
        try:
            self.info = core.read_bands_info(path)
        except Exception as e:
            messagebox.showerror("Could not read image", str(e))
            return
        self.input_path.set(path)
        if not self.output_path.get():
            self.output_path.set(os.path.splitext(path)[0] + "_prediction.tif")
        self._rebuild_bands()
        self._update_run_state()

    def _rebuild_bands(self):
        for child in self.bands_frame.winfo_children():
            child.destroy()
        self.combos = []
        if self.model is None or self.info is None:
            ttk.Label(self.bands_frame,
                      text="Select a model and an input image.").grid(
                          row=0, column=0, sticky="w")
            return

        count = self.info["count"]
        descriptions = self.info["descriptions"]
        options = [f"{b}: {descriptions[b - 1]}" if descriptions[b - 1] else f"Band {b}"
                   for b in range(1, count + 1)]
        note = ("counts match — identity default, adjust if needed"
                if count == self.model["in_channels"]
                else f"image has {count} band(s), model needs "
                     f"{self.model['in_channels']} — please map each channel")
        ttk.Label(self.bands_frame, text=note, foreground="#777").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        for i, name in enumerate(self.model["band_names"]):
            ttk.Label(self.bands_frame, text=f"ch {i + 1} · {name}").grid(
                row=i + 1, column=0, sticky="w", pady=1)
            combo = ttk.Combobox(self.bands_frame, values=options, state="readonly",
                                 width=28)
            combo.current(min(i, count - 1))  # identity default, clamped
            combo.bind("<<ComboboxSelected>>", lambda _e: self._update_run_state())
            combo.grid(row=i + 1, column=1, sticky="ew", pady=1)
            self.combos.append(combo)

    def _band_map(self):
        return [c.current() + 1 for c in self.combos]

    def _ready(self):
        return (self.model is not None and self.info is not None
                and bool(self.output_path.get())
                and self.combos and all(c.current() >= 0 for c in self.combos))

    def _update_run_state(self):
        self.run_btn["state"] = "normal" if self._ready() else "disabled"

    # ---------- run ----------

    def _run(self):
        if not self._ready():
            return
        self.run_btn["state"] = "disabled"
        self.progress["value"] = 0
        self.status["text"] = "Predicting…"
        threading.Thread(target=self._worker, daemon=True, args=(
            self.model_path.get(), self.input_path.get(),
            self.output_path.get(), self._band_map())).start()
        self.root.after(100, self._poll)

    def _worker(self, model_path, in_path, out_path, band_map):
        try:
            core.predict_geotiff(
                model_path, in_path, out_path, band_map=band_map,
                progress=lambda done, total: self.queue.put(("progress", done, total)))
            self.queue.put(("done", out_path))
        except Exception as e:  # any failure is reported to the user, not crashed
            self.queue.put(("error", str(e)))

    def _poll(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg[0] == "progress":
                    _, done, total = msg
                    self.progress["maximum"] = total
                    self.progress["value"] = done
                elif msg[0] == "done":
                    self.progress["value"] = self.progress["maximum"]
                    self.status["text"] = f"Done → {msg[1]}"
                    self.run_btn["state"] = "normal"
                    messagebox.showinfo("Prediction complete", f"Wrote:\n{msg[1]}")
                    return
                elif msg[0] == "error":
                    self.status["text"] = "Error"
                    self.run_btn["state"] = "normal"
                    messagebox.showerror("Prediction failed", msg[1])
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll)


def main():
    root = tk.Tk()
    PredictorGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    main()
