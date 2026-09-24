"""Desktop interface (customtkinter): pick a song, tick difficulties, press Generate.

Runs the pipeline in a worker thread and reports progress through a queue; all texts come from
`autoosu.i18n` so the window can switch between Chinese and English on the fly. Light and dark
looks share one palette taken from kanzei's OC: graphite black, amber-gold eyes and the electric-blue
cheek glow. The progress bar eases towards its target and the Generate button breathes while busy;
nothing else animates.
"""
from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

from . import __version__
from .difficulty import PRESETS
from .i18n import AUTHOR, language, set_language, tr
from .models import MODELS, app_root, ensure_model, find_model

from .audio_io import SUPPORTED_EXTS

AUDIO_EXT = tuple("*" + e for e in SUPPORTED_EXTS)
QUALITY_STEPS = {"fast": 50, "normal": 100, "high": 200}
ASSETS = Path(__file__).resolve().parent / "assets"
AVATAR = ASSETS / "avatar.png"          # kanzei's OC; header avatar + window icon when present

Pair = Tuple[str, str]                  # (light, dark)
PALETTE: Dict[str, Pair] = {
    "bg": ("#f4f3f0", "#0e1218"),
    "panel": ("#ffffff", "#171c24"),
    "inner": ("#e9e9ee", "#202024"),
    "border": ("#d6d6dc", "#2c2c32"),
    "text": ("#1c1c22", "#ecebe6"),
    "muted": ("#6f6f7a", "#8e8e98"),
    "gold": ("#e58a14", "#ffa329"),
    "gold_hover": ("#ca7510", "#e28a1d"),
    "cyan": ("#2b8fe6", "#4fb8ff"),      # cheek glow
    "warm": ("#e0a985", "#f2c9a0"),      # back-light on skin
    "warn": ("#b3541e", "#f0a060"),
}


def _hex(c: str) -> Tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def blend(a: str, b: str, t: float) -> str:
    """Colour between a (t=0) and b (t=1)."""
    t = max(0.0, min(1.0, t))
    return "#%02x%02x%02x" % tuple(int(round(x + (y - x) * t)) for x, y in zip(_hex(a), _hex(b)))


def settings_path() -> Path:
    override = os.environ.get("AUTOOSU_SETTINGS")          # tests / screenshots keep their own file
    if override:
        return Path(override)
    base = Path(os.environ.get("APPDATA") or Path.home())
    return base / "AUTO-OSU" / "settings.json"


def load_settings() -> Dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(data: Dict) -> None:
    try:
        settings_path().parent.mkdir(parents=True, exist_ok=True)
        settings_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def open_path(path: Path) -> None:
    """Open a file with its default app (an .osz opens in osu!) or a folder in Explorer."""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError:
        pass


def _avatar_image(size: int):
    """Round-cropped PIL image of assets/avatar.png, or None when the file is absent."""
    if not AVATAR.exists():
        return None
    try:
        from PIL import Image, ImageDraw

        img = Image.open(AVATAR).convert("RGBA")
        side = min(img.size)
        img = img.crop(((img.width - side) // 2, 0, (img.width - side) // 2 + side, side))
        big = size * 4
        img = img.resize((big, big), Image.LANCZOS)
        disc = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        ImageDraw.Draw(disc).ellipse((0, 0, big - 1, big - 1), fill=(36, 36, 40, 255))   # graphite backdrop
        disc.alpha_composite(img)
        mask = Image.new("L", (big, big), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, big - 1, big - 1), fill=255)
        disc.putalpha(mask)
        ImageDraw.Draw(disc).ellipse((2, 2, big - 3, big - 3), outline=(230, 169, 60, 255), width=max(4, big // 40))
        return disc.resize((size, size), Image.LANCZOS)
    except Exception:
        return None


def create_app():
    import customtkinter as ctk

    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
    except Exception:  # drag and drop is optional
        DND_FILES = TkinterDnD = None

    theme = ASSETS / "theme.json"
    ctk.set_default_color_theme(str(theme) if theme.exists() else "blue")
    ctk.set_appearance_mode("system")

    from .gui_layout import build_workspace

    class App(ctk.CTk):
        def __init__(self) -> None:
            super().__init__()
            self.dnd_ok = False
            if TkinterDnD is not None:
                try:
                    self.TkdndVersion = TkinterDnD._require(self)
                    self.dnd_ok = True
                except Exception:
                    self.dnd_ok = False
            self.settings = load_settings()
            self.generation_controls = self.settings.get("generation_controls", {})
            set_language(self.settings.get("language", "zh" if _system_is_chinese() else "en"))
            mode = self.settings.get("appearance") or "dark"
            self.mode = mode if mode in ("light", "dark") else "dark"
            ctk.set_appearance_mode(self.mode)
            self.widgets: Dict[str, object] = {}
            self.q: "queue.Queue" = queue.Queue()
            self.busy = False
            self.active_batch = False
            self.cancel_event = threading.Event()
            self.cuda_status = None
            self.cuda_checking = False
            self.managed_python = None
            self.runtime_installing = False
            self.runtime_cancel = threading.Event()
            self.jobs = []
            self.job_states = {}
            self.scan_token = 0
            self._scan_after = None
            self._closing = False
            self._open_after = False
            self.current_file = ""
            self.current_index = 0
            self.current_total = 0
            self.last_output_dir = None
            self.last_osz: Optional[Path] = None
            self.advanced_open = bool(self.settings.get("advanced_open", False))
            self._progress_target = 0.0
            self._progress_shown = 0.0
            self._pulsing = False
            self.font = ctk.CTkFont(family=_ui_font(), size=13)
            self.font_bold = ctk.CTkFont(family=_ui_font(), size=15, weight="bold")
            self.font_title = ctk.CTkFont(family=_ui_font(), size=25, weight="bold")
            self.geometry(self.settings.get("geometry", "1060x840"))
            self.minsize(940, 670)
            self.protocol("WM_DELETE_WINDOW", self.on_close)
            try:
                self.attributes("-alpha", 0.0)
            except tk.TclError:
                pass
            self.build()
            self.refresh_texts()
            self.refresh_models()
            self.after(100, self.poll)
            self.after(30, self._animate_progress)
            self.after(60, self._pulse)
            self.after(40, self._fade_in)
            self.after(120, self.check_cuda)
            self.after(150, self.schedule_scan)

        def c(self, key: str) -> str:
            return PALETTE[key][0 if self.mode == "light" else 1]

        # ------------------------------------------------------------------ layout
        def build(self) -> None:
            build_workspace(self, ctk, DND_FILES)

        def _set_window_icon(self) -> None:
            """Title-bar / taskbar icon: the bundled .ico (set after customtkinter installs its own)."""
            ico = ASSETS / "icon.ico"
            try:
                if ico.exists() and sys.platform == "win32":
                    self.iconbitmap(str(ico))
                    self.after(1000, lambda: self.iconbitmap(str(ico)))
                elif AVATAR.exists():
                    from PIL import Image, ImageTk

                    self._icon_img = ImageTk.PhotoImage(Image.open(AVATAR).convert("RGBA").resize((64, 64), Image.LANCZOS))
                    self.iconphoto(True, self._icon_img)
            except Exception:
                pass

        def refresh_texts(self) -> None:
            w = self.widgets
            if hasattr(self, "_quality_labels"):
                self.quality_var.set(next((k for k, v in self._quality_labels.items() if v == w["adv.quality.menu"].get()), "normal"))
                self.engine_var.set(next((k for k, v in self._engine_labels.items() if v == w["adv.engine.menu"].get()), "ml"))
            self.title(f"AUTO-OSU {__version__} · {AUTHOR}")
            self.header.set_texts(tr("app.title"), tr("app.subtitle"))
            self.header.lang_btn.configure(text=tr("lang.toggle"))
            self.header.theme_btn.configure(text=tr("theme.light" if self.mode == "dark" else "theme.dark"))
            for key in ("song.section", "song.browse", "diff.section", "diff.hint", "out.section",
                        "out.browse", "out.open_osu", "adv.seed", "adv.bpm", "adv.offset",
                        "adv.creator", "adv.star", "adv.quality", "adv.device", "adv.engine", "adv.preview",
                        "models.download", "run.open_osz", "run.open_folder", "run.check_source", "about", "input.recursive",
                        "input.queue", "batch.output_hint", "device.refresh", "batch.cancel",
                        "runtime.install", "runtime.hint", "runtime.cancel", "control.open"):
                w[key].configure(text=tr(key))
            for name in PRESETS:
                w[f"diff.{name}"].configure(text=tr(f"diff.{name}"))
            w["adv.toggle"].configure(text=tr("adv.hide" if self.advanced_open else "adv.show"))
            self._quality_labels = {k: tr(f"adv.quality.{k}") for k in QUALITY_STEPS}
            w["adv.quality.menu"].configure(values=list(self._quality_labels.values()))
            w["adv.quality.menu"].set(self._quality_labels.get(self.quality_var.get(), self._quality_labels["normal"]))
            self._engine_labels = {"ml": tr("adv.engine.ml"), "rules": tr("adv.engine.rules")}
            w["adv.engine.menu"].configure(values=list(self._engine_labels.values()))
            w["adv.engine.menu"].set(self._engine_labels.get(self.engine_var.get(), self._engine_labels["ml"]))
            if not self.busy:
                w["status"].configure(text=tr("status.idle"))
            self.refresh_models()
            self.update_input_mode()
            self.render_queue()
            self.refresh_cuda_text()
            self.refresh_control_summary()

        # ------------------------------------------------------------------ animations
        def _fade_in(self) -> None:
            try:
                a = float(self.attributes("-alpha"))
                if a < 1.0:
                    self.attributes("-alpha", min(1.0, a + 0.12))
                    self.after(16, self._fade_in)
            except tk.TclError:
                pass

        def set_progress(self, value: float) -> None:
            self._progress_target = max(0.0, min(1.0, value))
            if value <= 0.0:
                self._progress_shown = 0.0
                self.progress.set(0.0)

        def _animate_progress(self) -> None:
            try:
                d = self._progress_target - self._progress_shown
                if abs(d) > 0.0015:
                    self._progress_shown += d * 0.22
                    self.progress.set(self._progress_shown)
                elif d:
                    self._progress_shown = self._progress_target
                    self.progress.set(self._progress_shown)
            except tk.TclError:
                return
            self.after(30, self._animate_progress)

        def _pulse(self) -> None:
            """The Generate button breathes between amber and pale gold while work is running."""
            try:
                btn = self.widgets["run.generate"]
                if self.busy:
                    k = (math.sin(time.time() * 3.0) + 1) / 2
                    btn.configure(fg_color=blend(self.c("gold"), "#f7dc8f", 0.6 * k))   # gold -> pale gold, never greenish
                    self._pulsing = True
                elif self._pulsing:
                    btn.configure(fg_color=PALETTE["gold"])
                    self._pulsing = False
            except tk.TclError:
                return
            self.after(60, self._pulse)

        def _flash_done(self, n: int = 6) -> None:
            if n <= 0:
                self.progress.configure(progress_color=PALETTE["cyan"])
                return
            self.progress.configure(progress_color=PALETTE["gold"] if n % 2 else PALETTE["cyan"])
            self.after(140, lambda: self._flash_done(n - 1))

        def _fade_frame(self, frame, step: int = 0) -> None:
            if step > 6:
                frame.configure(fg_color=PALETTE["panel"])
                return
            frame.configure(fg_color=blend(self.c("bg"), self.c("panel"), step / 6))
            self.after(22, lambda: self._fade_frame(frame, step + 1))

        # ------------------------------------------------------------------ actions
        def toggle_language(self) -> None:
            set_language("en" if language() == "zh" else "zh")
            self.refresh_texts()

        def toggle_theme(self) -> None:
            self.mode = "light" if self.mode == "dark" else "dark"
            ctk.set_appearance_mode(self.mode)
            self.refresh_texts()

        def toggle_advanced(self) -> None:
            self.advanced_open = not self.advanced_open
            if self.advanced_open:
                self.adv_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))
                self._fade_frame(self.adv_frame)
            else:
                self.adv_frame.grid_forget()
            self.widgets["adv.toggle"].configure(text=tr("adv.hide" if self.advanced_open else "adv.show"))

        def browse_song(self) -> None:
            from tkinter import filedialog

            if self.source_mode.get() == "folder":
                path = filedialog.askdirectory(parent=self)
            else:
                path = filedialog.askopenfilename(parent=self, filetypes=[(tr("song.filetypes"), " ".join(AUDIO_EXT)), ("*", "*.*")])
            if path:
                self.song_var.set(path)

        def browse_out(self) -> None:
            from tkinter import filedialog

            path = filedialog.askdirectory(initialdir=self.out_var.get() or None)
            if path:
                self.out_var.set(path)

        def on_drop(self, event) -> None:
            if self.busy:
                return
            paths = self.tk.splitlist(event.data)
            if paths:
                self.source_mode.set("folder" if Path(paths[0]).is_dir() else "file")
                self.update_input_mode()
                self.song_var.set(paths[0])

        def change_input_mode(self, label) -> None:
            self.source_mode.set("folder" if label == tr("input.batch") else "file")
            self.update_input_mode()
            self.schedule_scan()

        def update_input_mode(self) -> None:
            folder = self.source_mode.get() == "folder"
            w = self.widgets
            w["input.mode"].configure(values=[tr("input.single"), tr("input.batch")])
            w["input.mode"].set(tr("input.batch" if folder else "input.single"))
            w["song.hint"].configure(text=tr("input.folder_hint" if folder else "song.hint"))
            for key in ("input.recursive", "batch.output_hint"):
                w[key].grid() if folder else w[key].grid_remove()
            w["out.open_osu"].grid_remove() if folder else w["out.open_osu"].grid()
            w["batch.cancel"].configure(state="normal" if self.busy and self.active_batch and not self.cancel_event.is_set() else "disabled")
            w["run.generate"].configure(text=tr("run.running" if self.busy else ("batch.start" if folder else "run.generate")))

        def schedule_scan(self, *_args) -> None:
            if self.busy:
                return
            self.scan_token += 1
            if self._scan_after:
                self.after_cancel(self._scan_after)
            self._scan_after = self.after(300, lambda token=self.scan_token: self.scan_source(token))

        def scan_source(self, token) -> None:
            self._scan_after = None
            raw = self.song_var.get().strip().strip('"')
            if not raw:
                self.jobs, self.job_states = [], {}
                self.render_queue()
                return
            source = Path(raw)
            destination = Path(self.out_var.get() or str(app_root()/"output")).resolve()
            recursive = self.recursive_var.get()
            self.widgets["input.count"].configure(text=tr("input.scan"))
            def scan():
                from .batch import scan_inputs
                try:
                    excluded = [destination] if destination != source.resolve() and destination.is_relative_to(source.resolve()) else []
                    jobs = scan_inputs(source, recursive, excluded)
                    self.q.put(("scan", token, jobs, source.is_dir(), None))
                except Exception as exc:
                    self.q.put(("scan", token, [], source.is_dir(), str(exc)))
            threading.Thread(target=scan, daemon=True).start()

        def render_queue(self, error=None) -> None:
            self.widgets["input.count"].configure(text=tr("input.count", n=len(self.jobs)))
            lines = []
            root = Path(self.song_var.get().strip().strip('"')).resolve()
            for job in self.jobs:
                state = self.job_states.get(str(job), "pending")
                path = Path(job)
                label = path.relative_to(root).as_posix() if self.source_mode.get() == "folder" and path.is_relative_to(root) else path.name
                lines.append(f"{tr('input.' + state):<10}  {label}")
            self.queue_box.configure(state="normal")
            self.queue_box.delete("1.0", "end")
            self.queue_box.insert("1.0", error or "\n".join(lines) or tr("input.empty"))
            self.queue_box.configure(state="disabled")
            if self.busy and self.current_index:
                self.queue_box.see(f"{self.current_index}.0")

        def check_cuda(self) -> None:
            if self.cuda_checking or self.busy:
                return
            self.cuda_checking = True
            self.widgets["device.refresh"].configure(state="disabled")
            self.widgets["run.generate"].configure(state="disabled")
            self.widgets["runtime.install"].configure(state="disabled")
            self.widgets["device.status"].configure(text=tr("device.checking"))
            def check():
                from .runtime import detect_runtime
                self.q.put(("cuda", detect_runtime()))
            threading.Thread(target=check, daemon=True).start()

        def refresh_cuda_text(self) -> None:
            status = self.cuda_status
            if status is None or self.cuda_checking:
                value = tr("device.checking")
            elif status.available:
                value = tr("device.ready", version=status.cuda_version, name=status.name, memory=status.memory_gb)
                if self.managed_python:
                    value += "\n" + tr("runtime.managed")
            else:
                value = tr("device." + status.reason, detail=status.detail)
                if status.reason == "cpu_build" and status.name:
                    value += "\n" + tr("device.gpu_found", name=status.name)
            self.widgets["device.status"].configure(text=value, text_color=PALETTE["cyan"] if status and status.available else PALETTE["muted"])
            self.widgets["runtime.install"].configure(text=tr("runtime.repair" if self.managed_python else "runtime.install"))

        def install_gpu_runtime(self) -> None:
            if self.busy or self.cuda_checking:
                return
            self.runtime_installing = True
            self.runtime_cancel.clear()
            self.cancel_event.clear()
            self.active_batch = False
            self.set_busy(True)
            self.set_progress(0)
            self.widgets["runtime.cancel"].configure(state="normal")
            self.widgets["runtime.cancel"].grid()
            self.widgets["status"].configure(text=tr("runtime.uv"))
            def install():
                from .runtime import SetupCancelled, install_runtime
                try:
                    result = install_runtime(log=lambda msg: self.q.put(("log", msg)),
                                             progress=lambda f, m: self.q.put(("progress", f, m)), cancel=self.runtime_cancel)
                    self.q.put(("runtime_ready", result))
                except SetupCancelled:
                    self.q.put(("runtime_cancelled", None))
                except Exception as exc:
                    self.q.put(("runtime_failed", str(exc)))
            threading.Thread(target=install, daemon=True).start()

        def cancel_runtime_setup(self) -> None:
            self.runtime_cancel.set()
            self.widgets["runtime.cancel"].configure(state="disabled")

        def cancel_batch(self) -> None:
            if self.busy and self.active_batch:
                self.cancel_event.set()
                self.widgets["batch.cancel"].configure(state="disabled", text=tr("batch.stopping"))
                self.widgets["status"].configure(text=tr("batch.stopping"))

        def open_last(self) -> None:
            if self.last_osz:
                open_path(self.last_osz)

        def open_output(self) -> None:
            path = self.last_output_dir or Path(self.out_var.get())
            path.mkdir(parents=True, exist_ok=True)
            open_path(path)

        def models_present(self) -> Dict[str, Optional[Path]]:
            return {name: find_model(name) for name in MODELS}

        def refresh_models(self) -> None:
            w = self.widgets
            found = self.models_present()
            if all(found.values()):
                w["models.status"].configure(text=tr("models.ok", rhythm=found["rhythm"].name, coord=found["coord"].name),
                                             text_color=PALETTE["text"])
                w["models.download"].grid_remove()
            else:
                w["models.status"].configure(text=tr("models.missing"), text_color=PALETTE["warn"])
                w["models.download"].grid()

        def download_models(self) -> None:
            if self.busy:
                return
            self.set_busy(True)

            def work() -> None:
                try:
                    for name in MODELS:
                        if not find_model(name):
                            self.q.put(("progress", 0.0, tr("models.downloading", name=MODELS[name].filename)))
                            ensure_model(name, lambda f, m: self.q.put(("progress", f, m)))
                    self.q.put(("models", None))
                except Exception as e:  # noqa: BLE001
                    self.q.put(("error", f"{e}"))

            threading.Thread(target=work, daemon=True).start()

        def log(self, text: str) -> None:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", text + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

        def set_busy(self, busy: bool) -> None:
            self.busy = busy
            state = "disabled" if busy else "normal"
            for key in ("run.generate", "models.download", "song.entry", "song.browse", "input.mode", "input.recursive",
                        "out.entry", "out.browse", "out.open_osu", "adv.device.menu", "adv.quality.menu", "adv.engine.menu", "adv.preview", "runtime.install", "control.open"):
                self.widgets[key].configure(state=state)
            for name in PRESETS:
                self.widgets[f"diff.{name}"].configure(state=state)
            for key in ("adv.seed", "adv.bpm", "adv.offset", "adv.creator", "adv.star"):
                self.widgets[key+".entry"].configure(state=state)
            self.widgets["device.refresh"].configure(state="disabled" if busy or self.cuda_checking else "normal")
            if self.cuda_checking:
                self.widgets["run.generate"].configure(state="disabled")
                self.widgets["runtime.install"].configure(state="disabled")
            self.widgets["batch.cancel"].configure(text=tr("batch.cancel"))
            self.update_input_mode()
            if self.runtime_installing:
                self.widgets["run.generate"].configure(text=tr("runtime.install"))

        def collect_settings(self) -> Dict:
            quality = next((k for k, v in self._quality_labels.items() if v == self.widgets["adv.quality.menu"].get()), "normal")
            engine = next((k for k, v in self._engine_labels.items() if v == self.widgets["adv.engine.menu"].get()), "ml")
            self.quality_var.set(quality)
            self.engine_var.set(engine)
            return {
                "language": language(), "appearance": self.mode, "song": self.song_var.get(), "out_dir": self.out_var.get(),
                "difficulties": [n for n, v in self.diff_vars.items() if v.get()],
                "open_osu": self.open_osu_var.get(), "seed": self.seed_var.get(), "bpm": self.bpm_var.get(),
                "offset": self.offset_var.get(), "creator": self.creator_var.get(), "star": self.star_var.get(),
                "quality": quality, "device": self.device_var.get(), "engine": engine,
                "preview": self.preview_var.get(), "advanced_open": self.advanced_open, "geometry": self.geometry(),
                "source_mode": self.source_mode.get(), "recursive": self.recursive_var.get(),
                "generation_controls": self.generation_controls,
            }

        def edit_generation_controls(self):
            if self.busy:
                return
            from .control_gui import open_control_window
            if getattr(self, "control_window", None) and self.control_window.winfo_exists():
                self.control_window.lift()
            else:
                self.control_window = open_control_window(self)

        def refresh_control_summary(self):
            from .controls import validate_controls
            try:
                controls = validate_controls(self.generation_controls)
            except ValueError:
                self.widgets["control.status"].configure(text=tr("control.invalid"))
                return
            parts = []
            for key in ("target_stars", "density", "spacing_scale"):
                if controls[key] is not None:
                    parts.append(tr("control."+key)+f": {controls[key]:g}")
            if controls["density_curve"]:
                parts.append(tr("control.curve_active"))
            if controls["skill_preference"] != "balanced":
                parts.append(tr("control.preference."+controls["skill_preference"]))
            if controls["highlight_mode"] != "legacy":
                parts.append(tr("control.mode."+controls["highlight_mode"]))
            self.widgets["control.status"].configure(text=" · ".join(parts) or tr("control.default"))

        def log_preference(self, summary):
            preference = summary.get("controls", {}).get("preference")
            if preference:
                self.log(tr("control.result.preference", name=tr("control.preference."+preference["requested"]),
                            result=tr("control.result.preference_"+preference["status"])))

        def start(self) -> None:
            if self.busy or self.cuda_checking:
                return
            s = self.collect_settings()
            save_settings(s)
            song = Path(s["song"].strip().strip('"'))
            if not s["song"].strip():
                return self.show_error(tr("err.no_song"))
            if not song.exists():
                return self.show_error(tr("err.file_missing", path=song))
            if s["source_mode"] == "folder" and not song.is_dir():
                return self.show_error(tr("input.folder_hint"))
            if not s["difficulties"]:
                return self.show_error(tr("err.no_diff"))
            found = self.models_present()
            if s["engine"] == "ml" and not all(found.values()):
                return self.show_error(tr("err.no_models"))

            def num(v: str):
                v = v.strip()
                return float(v) if v else None

            try:
                from .controls import validate_controls
                controls = validate_controls(s["generation_controls"])
                seed = int(s["seed"] or 0)
                bpm, offset, star = num(s["bpm"]), num(s["offset"]), num(s["star"])
            except ValueError as e:
                return self.show_error(str(e))
            self.log_box.configure(state="normal")
            self.log_box.delete("1.0", "end")
            self.log_box.configure(state="disabled")
            self.set_progress(0.0)
            self.widgets["run.open_osz"].configure(state="disabled")
            self.last_osz = None
            self.scan_token += 1
            self.cancel_event.clear()
            self.active_batch = song.is_dir()
            self.source_mode.set("folder" if self.active_batch else "file")
            self._open_after = s["open_osu"] and not self.active_batch
            self.last_output_dir = Path(s["out_dir"] or str(app_root()/"output")).resolve()
            self.job_states = {}
            if not self.active_batch:
                self.jobs = [song.resolve()]
                self.job_states[str(song.resolve())] = "running"
            self.render_queue()
            self.set_busy(True)
            kwargs = dict(
                seed=seed, bpm=bpm, offset_ms=offset, creator=s["creator"] or "AUTO-OSU", star_rating=star,
                coord_steps=QUALITY_STEPS[s["quality"]], device=s["device"], controls=controls,
                rhythm_model=str(found["rhythm"]) if s["engine"] == "ml" else None,
                coord_model=str(found["coord"]) if s["engine"] == "ml" else None,
            )
            threading.Thread(target=self.work, args=(song, s["difficulties"], self.last_output_dir, kwargs, s["preview"], s["recursive"]),
                             daemon=True).start()

        def work(self, song: Path, diffs, out_dir: Path, kwargs: Dict, preview: bool, recursive: bool = False) -> None:
            from .generate import generate

            try:
                if self.managed_python and kwargs.get("device") != "cpu" and kwargs.get("rhythm_model"):
                    from .runtime import run_in_runtime
                    request = dict(source=str(song.resolve()), difficulties=diffs, out_dir=str(out_dir.resolve()),
                                   kwargs=kwargs, preview=preview, recursive=recursive)
                    run_in_runtime(self.managed_python, request, self.worker_event, self.cancel_event)
                    return
                if song.is_dir():
                    from .batch import generate_batch
                    def notify(i, n, item):
                        self.q.put(("batch_item", i, n, item.source, item.status, item.osz, item.error))
                    result = generate_batch(song, diffs, out_dir, recursive=recursive, preview=preview,
                                            cancel=self.cancel_event, on_item=notify,
                                            log=lambda t: self.q.put(("log", t)),
                                            progress=lambda f, m: self.q.put(("progress", f, m)), **kwargs)
                    self.q.put(("batch_done", result))
                    return
                res = generate(song, diffs, out_dir, log=lambda t: self.q.put(("log", t)),
                               progress=lambda f, m: self.q.put(("progress", f, m)), **kwargs)
                if preview:
                    from .preview import render_preview

                    for d in res.diffs:
                        out = out_dir / f"{song.stem} [{d.preset.name}]_preview.mp3"
                        self.q.put(("log", f"preview: {render_preview(res.audio_file, d.beatmap, out, click_shift_ms=res.osu_shift_ms)}"))
                from .worker import summary
                self.q.put(("single_done", summary(res)))
            except Exception as e:  # noqa: BLE001
                self.q.put(("log", traceback.format_exc()))
                self.q.put(("error", f"{type(e).__name__}: {e}"))

        def worker_event(self, event) -> None:
            kind = event["event"]
            if kind == "log":
                self.q.put(("log", event["text"]))
            elif kind == "progress":
                self.q.put(("progress", event["fraction"], event["message"]))
            elif kind == "single_done":
                self.q.put(("single_done", event["result"]))
            elif kind == "batch_item":
                job = event["item"]
                self.q.put(("batch_item", event["index"], event["total"], job["source"], job["status"], job["osz"], job["error"]))
            elif kind == "batch_done":
                from .batch import BatchItem, BatchResult
                self.q.put(("batch_done", BatchResult(Path(event["report"]), [BatchItem(**item) for item in event["items"]],
                                                       event["elapsed_s"], event["cancelled"])))

        def status_text(self, msg: str) -> str:
            if msg in _keys():
                return tr(msg)
            if ": rhythm" in msg:
                return tr("status.rhythm", diff=msg.split(":")[0])
            if ": placing" in msg:
                diff, _, detail = msg.partition(": placing")
                return tr("status.placing", diff=diff, detail=detail.strip())
            return tr(f"status.{msg}") if f"status.{msg}" in _keys() else msg

        def poll(self) -> None:
            try:
                while True:
                    item = self.q.get_nowait()
                    kind = item[0]
                    if kind == "cuda":
                        self.cuda_status, self.managed_python = item[1].cuda, item[1].python
                        self.cuda_checking = False
                        self.widgets["device.refresh"].configure(state="disabled" if self.busy else "normal")
                        self.widgets["run.generate"].configure(state="disabled" if self.busy else "normal")
                        self.widgets["runtime.install"].configure(state="disabled" if self.busy else "normal")
                        if item[1].managed_error:
                            self.log(item[1].managed_error)
                        self.refresh_cuda_text()
                    elif kind in ("runtime_ready", "runtime_cancelled", "runtime_failed"):
                        self.runtime_installing = False
                        self.widgets["runtime.cancel"].grid_remove()
                        self.set_busy(False)
                        if kind == "runtime_ready":
                            self.cuda_status, self.managed_python = item[1].cuda, item[1].python
                            self.set_progress(1)
                            self.device_var.set("auto")
                            self.widgets["status"].configure(text=tr("runtime.ready"))
                            self.refresh_cuda_text()
                        else:
                            self.set_progress(0)
                            self.widgets["status"].configure(text=tr("runtime.cancelled" if kind == "runtime_cancelled" else "runtime.failed"))
                            if item[1]:
                                self.log(item[1])
                    elif kind == "scan":
                        if item[1] == self.scan_token and not self.busy:
                            self.jobs, self.job_states = item[2], {}
                            if item[3]:
                                self.source_mode.set("folder")
                                self.update_input_mode()
                            self.render_queue(item[4])
                    elif kind == "batch_item":
                        _, i, total, source, state, osz, error = item
                        self.current_index, self.current_total = i + 1, total
                        self.current_file = Path(source).name
                        if Path(source) not in self.jobs:
                            self.jobs.append(Path(source))
                        self.job_states[source] = state
                        if osz:
                            self.last_osz = Path(osz)
                        self.render_queue()
                    elif kind == "batch_done":
                        result = item[1]
                        self.set_busy(False)
                        self.jobs = [Path(job.source) for job in result.items]
                        self.job_states = {job.source: job.status for job in result.items}
                        self.render_queue()
                        remaining = sum(job.status == "cancelled" for job in result.items)
                        self.set_progress((result.succeeded + result.failed) / len(result.items))
                        key = "batch.stopped" if result.cancelled else "batch.done"
                        self.widgets["status"].configure(text=tr(key, ok=result.succeeded, failed=result.failed,
                                                               remaining=remaining, secs=result.elapsed_s))
                        self.widgets["run.open_osz"].configure(state="normal" if self.last_osz else "disabled")
                        self.log(tr("batch.report", path=result.report))
                        for job in result.items:
                            for summary in job.diffs:
                                if summary.get("controls", {}).get("preference"):
                                    self.log(f"{Path(job.source).name} [{summary['name']}]")
                                    self.log_preference(summary)
                        self.active_batch = False
                    elif kind == "log":
                        self.log(item[1])
                    elif kind == "progress":
                        self.set_progress(item[1])
                        if not self.cancel_event.is_set():
                            if item[2] != "batch item done":
                                prefix = f"{self.current_index}/{self.current_total} · {self.current_file} — " if self.active_batch else ""
                                self.widgets["status"].configure(text=prefix + self.status_text(item[2]))
                    elif kind == "models":
                        self.set_busy(False)
                        self.set_progress(0.0)
                        self.widgets["status"].configure(text=tr("status.idle"))
                        self.refresh_models()
                    elif kind == "single_done":
                        res = item[1]
                        self.set_busy(False)
                        self.set_progress(1.0)
                        self.last_osz = Path(res["osz"])
                        if self.jobs:
                            self.job_states[str(self.jobs[0])] = "done"
                            self.render_queue()
                        self.widgets["run.open_osz"].configure(state="normal")
                        self.widgets["status"].configure(
                            text=tr("status.done", file=self.last_osz.name, secs=res["elapsed_s"], device=res["device"]))
                        self.log(tr("result.summary", bpm=res["bpm"], n=len(res["diffs"])))
                        for s in res["diffs"]:
                            self.log(tr("result.diff", name=s["name"], objects=s["objects"], sliders=s["sliders"], nps=s["nps"]))
                            if s.get("measured_stars") is not None:
                                self.log(tr("result.stars", stars=s["measured_stars"]))
                            else:
                                self.log(tr("result.stars_unavailable"))
                            if s.get("star_condition") is not None:
                                self.log(tr("result.condition", stars=s["star_condition"]))
                            if s.get("watermark", {}).get("status") in ("embedded", "insufficient", "unsupported"):
                                self.log(tr("result.watermark." + s["watermark"]["status"]))
                            control = s.get("controls", {})
                            self.log_preference(s)
                            if control.get("target_stars") is not None:
                                self.log(tr("control.result.target", target=control["target_stars"],
                                            result=tr("control.result.met" if control["target_met"] else "control.result.missed"),
                                            count=len(control["candidates"])))
                            plan = control.get("highlight_plan", {})
                            if plan.get("mode") in ("manual", "auto"):
                                regions = plan.get("regions", [])
                                self.log(tr("control.result.regions", regions=", ".join(f"{r['start_s']:.1f}–{r['end_s']:.1f}s" for r in regions) or tr("control.result.none")))
                        if res.get("evaluation_path"):
                            self.log(tr("result.evaluation", path=res["evaluation_path"]))
                        self._flash_done()
                        if self._open_after and not self._closing:
                            open_path(self.last_osz)
                    elif kind == "error":
                        self.set_busy(False)
                        for key, value in self.job_states.items():
                            if value == "running":
                                self.job_states[key] = "error"
                        self.render_queue()
                        self.widgets["status"].configure(text=tr("status.error", err=item[1]))
                        if not self._closing:
                            self.show_error(item[1])
            except queue.Empty:
                pass
            if self._closing and not self.busy:
                self.destroy()
                return
            self.after(100, self.poll)

        def show_error(self, text: str) -> None:
            from tkinter import messagebox

            messagebox.showerror("AUTO-OSU", text)

        def check_source(self) -> None:
            from .source_gui import open_source_window

            existing = getattr(self, "source_window", None)
            if existing is not None and existing.winfo_exists():
                existing.lift()
            else:
                self.source_window = open_source_window(self)

        def on_close(self) -> None:
            try:
                save_settings(self.collect_settings())
            except Exception:
                pass
            if self.busy:
                self._closing = True
                self.cancel_event.set()
                self.runtime_cancel.set()
                self.widgets["status"].configure(text=tr("batch.stopping"))
            else:
                self.destroy()

    return App()


def run_gui() -> int:
    app = create_app()
    app.mainloop()
    return 0


def _keys():
    from .i18n import STRINGS

    return STRINGS


def _system_is_chinese() -> bool:
    try:
        import locale

        lang = locale.getlocale()[0] or ""
        if not lang and sys.platform == "win32":
            import ctypes

            lang = locale.windows_locale.get(ctypes.windll.kernel32.GetUserDefaultUILanguage(), "")
        return lang.lower().startswith(("zh", "chinese"))
    except Exception:
        return False


FONTS_DIR = ASSETS / "fonts"
_FONT_FAMILY: Optional[str] = None


def _register_fonts() -> Optional[str]:
    """Load the bundled UI font for this process (Windows) and return its family name, else None."""
    global _FONT_FAMILY
    if _FONT_FAMILY is not None:
        return _FONT_FAMILY or None
    _FONT_FAMILY = ""
    if sys.platform == "win32" and FONTS_DIR.is_dir():
        try:
            import ctypes

            n = 0
            for f in sorted(FONTS_DIR.glob("*.ttf")):
                n += ctypes.windll.gdi32.AddFontResourceExW(str(f), 0x10, 0)   # FR_PRIVATE
            if n:
                _FONT_FAMILY = "AUTO-OSU Sans"
        except Exception:
            _FONT_FAMILY = ""
    return _FONT_FAMILY or None


def _ui_font() -> str:
    fam = _register_fonts()
    if fam:
        return fam
    if sys.platform == "win32":
        try:
            import tkinter.font as tkfont

            fams = set(tkfont.families())
            for cand in ("Segoe UI Variable Text", "Segoe UI", "Microsoft YaHei UI"):
                if cand in fams:
                    return cand
        except Exception:
            pass
        return "Segoe UI"
    return "Helvetica"


if __name__ == "__main__":
    sys.exit(run_gui())
