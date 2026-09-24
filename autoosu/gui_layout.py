"""The converter workspace: illustrated cover, file queue and compact settings."""
from __future__ import annotations

import tkinter as tk

from .difficulty import PRESETS
from .i18n import tr
from .models import app_root


class Header:
    def __init__(self, app, master, ctk):
        from .gui import ASSETS
        from PIL import Image

        self.app = app
        self.frame = ctk.CTkFrame(master, height=182, corner_radius=0, fg_color="#10141b")
        self.frame.grid(row=0, column=0, sticky="ew")
        self.frame.grid_propagate(False)
        self.canvas = tk.Canvas(self.frame, bg="#10141b", highlightthickness=0)
        self.canvas.place(x=0, y=0, relwidth=1, relheight=1)
        path = ASSETS / "cover.png"
        self.source = Image.open(path).convert("RGB") if path.exists() else None
        self.subtitle = ""
        self.canvas.bind("<Configure>", self.redraw)
        kw = dict(width=84, height=27, font=app.font, fg_color="#242933", hover_color="#333b47",
                  text_color="#f1f0e9", corner_radius=6)
        self.lang_btn = ctk.CTkButton(self.frame, command=app.toggle_language, **kw)
        self.lang_btn.place(x=28, y=138)
        self.theme_btn = ctk.CTkButton(self.frame, command=app.toggle_theme, **kw)
        self.theme_btn.place(x=120, y=138)

    def set_texts(self, title, subtitle):
        self.subtitle = subtitle
        self.redraw()

    def redraw(self, event=None):
        from PIL import Image, ImageTk
        from .gui import _ui_font

        c = self.canvas
        width, height = c.winfo_width(), c.winfo_height()
        if width < 2 or height < 2:
            return
        c.delete("all")
        scale = self.frame._get_widget_scaling()
        if self.source:
            iw = round(height * self.source.width / self.source.height)
            self.photo = ImageTk.PhotoImage(self.source.resize((iw, height), Image.Resampling.LANCZOS))
            c.create_image(width, 0, image=self.photo, anchor="ne")
        def label(x, y, value, size, color, weight="normal"):
            c.create_text(x*scale, y*scale, text=value, anchor="nw", fill=color,
                          font=(_ui_font(), -round(size*scale), weight))
        label(29, 16, "BY KANZEI   /   BEATMAP CONVERTER", 10, "#adb4c0")
        label(26, 38, "AUTO-OSU", 40, "#fff6e7", "bold")
        label(29, 98, self.subtitle, 14, "#e2e5e9")


def build_workspace(app, ctk, dnd_files):
    from .gui import PALETTE, QUALITY_STEPS, _ui_font

    a, w, s = app, app.widgets, app.settings
    a.grid_columnconfigure(0, weight=1)
    a.grid_rowconfigure(1, weight=1)
    a.header = Header(a, a, ctk)
    a.after(400, a._set_window_icon)
    a.body = body = ctk.CTkScrollableFrame(a, fg_color="transparent", corner_radius=0)
    body.grid(row=1, column=0, sticky="nsew", padx=22, pady=(14, 0))
    body.grid_columnconfigure(0, weight=3, minsize=440)
    body.grid_columnconfigure(1, weight=2, minsize=295)
    left = ctk.CTkFrame(body, fg_color="transparent")
    left.grid(row=0, column=0, sticky="new", padx=(0, 27))
    left.grid_columnconfigure(0, weight=1)
    right = ctk.CTkFrame(body, fg_color="transparent")
    right.grid(row=0, column=1, sticky="new", padx=(16, 8))
    right.grid_columnconfigure(0, weight=1)

    a.song_var = ctk.StringVar(value=s.get("song", ""))
    a.source_mode = ctk.StringVar(value=s.get("source_mode", "file"))
    a.recursive_var = ctk.BooleanVar(value=s.get("recursive", False))
    a.out_var = ctk.StringVar(value=s.get("out_dir", str(app_root()/"output")))
    a.open_osu_var = ctk.BooleanVar(value=s.get("open_osu", True))
    a.device_var = ctk.StringVar(value=s.get("device", "auto"))
    a.seed_var = ctk.StringVar(value=str(s.get("seed", 0)))
    a.bpm_var = ctk.StringVar(value=s.get("bpm", ""))
    a.offset_var = ctk.StringVar(value=s.get("offset", ""))
    a.creator_var = ctk.StringVar(value=s.get("creator", "AUTO-OSU"))
    a.star_var = ctk.StringVar(value=s.get("star", ""))
    a.quality_var = ctk.StringVar(value=s.get("quality", "normal"))
    a.engine_var = ctk.StringVar(value=s.get("engine", "ml"))
    a.preview_var = ctk.BooleanVar(value=s.get("preview", False))

    w["song.section"] = ctk.CTkLabel(left, font=a.font_bold, anchor="w")
    w["song.section"].grid(row=0, column=0, sticky="w")
    w["input.mode"] = ctk.CTkSegmentedButton(left, font=a.font, height=34, values=[tr("input.single"), tr("input.batch")], command=a.change_input_mode)
    w["input.mode"].grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 10))
    w["song.entry"] = ctk.CTkEntry(left, textvariable=a.song_var, font=a.font, height=36)
    w["song.entry"].grid(row=2, column=0, sticky="ew", padx=(0, 8))
    w["song.browse"] = ctk.CTkButton(left, width=88, height=36, font=a.font, command=a.browse_song)
    w["song.browse"].grid(row=2, column=1)
    w["song.hint"] = ctk.CTkLabel(left, font=a.font, anchor="w", justify="left", wraplength=480, text_color=PALETTE["muted"])
    w["song.hint"].grid(row=3, column=0, columnspan=2, sticky="w", pady=(7, 3))
    w["input.recursive"] = ctk.CTkCheckBox(left, variable=a.recursive_var, font=a.font, command=a.schedule_scan, height=24)
    w["input.recursive"].grid(row=4, column=0, columnspan=2, sticky="w", pady=4)
    queue_head = ctk.CTkFrame(left, fg_color="transparent")
    queue_head.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(13, 4))
    queue_head.grid_columnconfigure(0, weight=1)
    w["input.queue"] = ctk.CTkLabel(queue_head, font=a.font_bold, anchor="w")
    w["input.queue"].grid(row=0, column=0, sticky="w")
    w["input.count"] = ctk.CTkLabel(queue_head, font=a.font, text_color=PALETTE["muted"])
    w["input.count"].grid(row=0, column=1, sticky="e")
    a.queue_box = ctk.CTkTextbox(left, font=a.font, height=124, corner_radius=6, border_width=1, border_color=PALETTE["border"])
    a.queue_box.grid(row=6, column=0, columnspan=2, sticky="ew")
    a.queue_box.configure(state="disabled")
    w["out.section"] = ctk.CTkLabel(left, font=a.font_bold, anchor="w")
    w["out.section"].grid(row=7, column=0, sticky="w", pady=(16, 4))
    w["out.entry"] = ctk.CTkEntry(left, textvariable=a.out_var, font=a.font, height=34)
    w["out.entry"].grid(row=8, column=0, sticky="ew", padx=(0, 8))
    w["out.browse"] = ctk.CTkButton(left, width=88, height=34, font=a.font, command=a.browse_out)
    w["out.browse"].grid(row=8, column=1)
    w["out.open_osu"] = ctk.CTkCheckBox(left, variable=a.open_osu_var, font=a.font)
    w["out.open_osu"].grid(row=9, column=0, columnspan=2, sticky="w", pady=(9, 4))
    w["batch.output_hint"] = ctk.CTkLabel(left, font=a.font, text_color=PALETTE["muted"], wraplength=480, justify="left", anchor="w")
    w["batch.output_hint"].grid(row=10, column=0, columnspan=2, sticky="w", pady=(4, 0))

    if a.dnd_ok:
        for target in (left, w["song.entry"], w["song.hint"], a.queue_box):
            try:
                target.drop_target_register(dnd_files)
                target.dnd_bind("<<Drop>>", a.on_drop)
            except Exception:
                pass
    w["diff.section"] = ctk.CTkLabel(right, font=a.font_bold, anchor="w")
    w["diff.section"].grid(row=0, column=0, sticky="w")
    a.diff_vars = {}
    chosen = set(s.get("difficulties", ["Hard", "Insane"]))
    diff_options = ctk.CTkFrame(right, fg_color="transparent")
    diff_options.grid(row=1, column=0, sticky="ew")
    diff_options.grid_columnconfigure((0, 1), weight=1)
    for i, name in enumerate(PRESETS):
        var = ctk.BooleanVar(value=name in chosen)
        a.diff_vars[name] = var
        w[f"diff.{name}"] = ctk.CTkCheckBox(diff_options, variable=var, font=a.font, height=27, width=135)
        w[f"diff.{name}"].grid(row=i//2, column=i%2, sticky="w", pady=(8, 0))
    w["diff.hint"] = ctk.CTkLabel(right, font=a.font, anchor="w", justify="left", wraplength=305, text_color=PALETTE["muted"])
    w["diff.hint"].grid(row=5, column=0, sticky="w", pady=(8, 16))
    device_head = ctk.CTkFrame(right, fg_color="transparent")
    device_head.grid(row=6, column=0, sticky="ew")
    device_head.grid_columnconfigure(0, weight=1)
    w["adv.device"] = ctk.CTkLabel(device_head, font=a.font_bold, anchor="w")
    w["adv.device"].grid(row=0, column=0, sticky="w")
    w["device.refresh"] = ctk.CTkButton(device_head, width=90, height=25, font=a.font, text_color=PALETTE["gold"],
                                      fg_color="transparent", border_width=1, border_color=PALETTE["border"], command=a.check_cuda)
    w["device.refresh"].grid(row=0, column=1)
    w["adv.device.menu"] = ctk.CTkOptionMenu(right, variable=a.device_var, values=["auto", "cuda", "cpu"], font=a.font, height=32)
    w["adv.device.menu"].grid(row=7, column=0, sticky="ew", pady=(6, 5))
    w["device.status"] = ctk.CTkLabel(right, font=a.font, anchor="w", justify="left", wraplength=325, text_color=PALETTE["muted"])
    w["device.status"].grid(row=8, column=0, sticky="w")
    w["models.status"] = ctk.CTkLabel(right, font=a.font, anchor="w", justify="left", wraplength=325)
    w["models.status"].grid(row=9, column=0, sticky="w", pady=(14, 3))
    w["models.download"] = ctk.CTkButton(right, height=30, font=a.font, command=a.download_models)
    w["models.download"].grid(row=10, column=0, sticky="w")
    w["runtime.install"] = ctk.CTkButton(right, height=33, font=a.font, command=a.install_gpu_runtime)
    w["runtime.install"].grid(row=11, column=0, sticky="ew", pady=(13, 4))
    w["runtime.hint"] = ctk.CTkLabel(right, font=a.font, text_color=PALETTE["muted"], anchor="w", justify="left", wraplength=325)
    w["runtime.hint"].grid(row=12, column=0, sticky="w")
    w["runtime.cancel"] = ctk.CTkButton(right, font=a.font, fg_color="transparent", border_width=1,
                                      border_color=PALETTE["border"], text_color=PALETTE["text"], command=a.cancel_runtime_setup)
    w["runtime.cancel"].grid(row=13, column=0, sticky="w", pady=4)
    w["runtime.cancel"].grid_remove()

    w["adv.toggle"] = ctk.CTkButton(body, fg_color="transparent", anchor="w", font=a.font, text_color=PALETTE["gold"],
                                    hover=False, command=a.toggle_advanced)
    w["adv.toggle"].grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 2))
    a.adv_frame = adv = ctk.CTkFrame(body, fg_color=PALETTE["panel"], corner_radius=8)
    adv.grid_columnconfigure((1, 3), weight=1)
    for i, (key, var) in enumerate([("adv.seed", a.seed_var), ("adv.bpm", a.bpm_var), ("adv.offset", a.offset_var),
                                   ("adv.creator", a.creator_var), ("adv.star", a.star_var)]):
        r, col = divmod(i, 2)
        w[key] = ctk.CTkLabel(adv, font=a.font, anchor="w")
        w[key].grid(row=r, column=2*col, sticky="w", padx=(12, 6), pady=5)
        w[key+".entry"] = ctk.CTkEntry(adv, textvariable=var, font=a.font, width=125)
        w[key+".entry"].grid(row=r, column=2*col+1, sticky="ew", padx=(0, 12), pady=5)
    w["adv.quality"] = ctk.CTkLabel(adv, font=a.font, anchor="w")
    w["adv.quality"].grid(row=3, column=0, sticky="w", padx=12, pady=5)
    w["adv.quality.menu"] = ctk.CTkOptionMenu(adv, font=a.font, values=["Standard"], width=175)
    w["adv.quality.menu"].grid(row=3, column=1, sticky="w", pady=5)
    w["adv.engine"] = ctk.CTkLabel(adv, font=a.font, anchor="w")
    w["adv.engine"].grid(row=3, column=2, sticky="w", padx=12, pady=5)
    w["adv.engine.menu"] = ctk.CTkOptionMenu(adv, font=a.font, values=["AI"], width=235)
    w["adv.engine.menu"].grid(row=3, column=3, sticky="w", padx=(0, 12), pady=5)
    w["adv.preview"] = ctk.CTkCheckBox(adv, variable=a.preview_var, font=a.font)
    w["adv.preview"].grid(row=4, column=0, columnspan=4, sticky="w", padx=12, pady=(5, 12))
    w["control.open"] = ctk.CTkButton(adv, font=a.font, command=a.edit_generation_controls)
    w["control.open"].grid(row=5, column=0, columnspan=4, sticky="ew", padx=12, pady=(0, 12))
    if a.advanced_open:
        adv.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))

    run = ctk.CTkFrame(a, fg_color=PALETTE["panel"], corner_radius=0)
    run.grid(row=2, column=0, sticky="ew")
    run.grid_columnconfigure(0, weight=1)
    a.progress = ctk.CTkProgressBar(run, height=4, corner_radius=0, progress_color=PALETTE["gold"])
    a.progress.grid(row=0, column=0, sticky="ew")
    a.progress.set(0)
    w["status"] = ctk.CTkLabel(run, font=a.font, anchor="w", justify="left", wraplength=940)
    w["status"].grid(row=1, column=0, sticky="w", padx=26, pady=(7, 2))
    w["control.status"] = ctk.CTkLabel(run, font=a.font, anchor="w", justify="left", wraplength=940,
                                      text_color=PALETTE["muted"])
    w["control.status"].grid(row=2, column=0, sticky="w", padx=26, pady=(0, 3))
    buttons = ctk.CTkFrame(run, fg_color="transparent")
    buttons.grid(row=3, column=0, sticky="ew", padx=26, pady=(3, 9))
    w["run.generate"] = ctk.CTkButton(buttons, height=38, width=185, font=a.font_bold, command=a.start)
    w["run.generate"].pack(side="left")
    secondary = dict(height=38, width=140, font=a.font, fg_color="transparent", border_width=1,
                     border_color=PALETTE["border"], text_color=PALETTE["text"])
    w["batch.cancel"] = ctk.CTkButton(buttons, command=a.cancel_batch, **secondary)
    w["batch.cancel"].pack(side="left", padx=(8, 0))
    w["run.open_osz"] = ctk.CTkButton(buttons, command=lambda: a.last_osz and a.open_last(), state="disabled", **secondary)
    w["run.open_osz"].pack(side="left", padx=8)
    w["run.open_folder"] = ctk.CTkButton(buttons, command=a.open_output, **secondary)
    w["run.open_folder"].pack(side="left")
    w["run.check_source"] = ctk.CTkButton(buttons, command=a.check_source, **secondary)
    w["run.check_source"].pack(side="left", padx=(8, 0))
    a.log_box = ctk.CTkTextbox(run, font=ctk.CTkFont(family="Consolas", size=11), height=62, corner_radius=4)
    a.log_box.grid(row=4, column=0, sticky="ew", padx=26, pady=(0, 7))
    a.log_box.configure(state="disabled")
    w["about"] = ctk.CTkLabel(run, font=ctk.CTkFont(family=_ui_font(), size=10), text_color=PALETTE["muted"])
    w["about"].grid(row=5, column=0, sticky="e", padx=26, pady=(0, 6))
    a.song_var.trace_add("write", a.schedule_scan)
    a.out_var.trace_add("write", a.schedule_scan)
