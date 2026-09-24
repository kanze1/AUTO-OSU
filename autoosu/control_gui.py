"""Small optional-control editor shared by single and folder generation."""
from __future__ import annotations

from tkinter import filedialog, messagebox
from pathlib import Path

from .controls import load_control_file, validate_controls
from .i18n import STRINGS, language
from .provenance import atomic_json


def open_control_window(parent):
    import customtkinter as ctk
    lang = language()
    def text(key):
        return STRINGS["control."+key][lang]
    window = ctk.CTkToplevel(parent)
    window.title(text("title"))
    window.geometry("740x640")
    window.minsize(660, 540)
    window.grid_columnconfigure(0, weight=1)
    window.grid_rowconfigure(0, weight=1)
    body = ctk.CTkScrollableFrame(window, fg_color="transparent")
    body.grid(row=0, column=0, sticky="nsew", padx=16, pady=12)
    body.grid_columnconfigure(1, weight=1)
    ctk.CTkLabel(body, text=text("hint"), wraplength=650, justify="left", font=parent.font).grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))
    fields = {}
    for row, key in enumerate(("target_stars", "density", "spacing_scale", "candidates"), 1):
        ctk.CTkLabel(body, text=text(key), font=parent.font).grid(row=row, column=0, sticky="w", padx=(0, 14), pady=5)
        fields[key] = ctk.StringVar()
        ctk.CTkEntry(body, textvariable=fields[key], font=parent.font).grid(row=row, column=1, sticky="ew", pady=5)
    labels = {mode: text("mode."+mode) for mode in ("auto", "manual", "off", "legacy")}
    ctk.CTkLabel(body, text=text("mode"), font=parent.font).grid(row=5, column=0, sticky="w", pady=8)
    mode_var = ctk.StringVar()
    ctk.CTkOptionMenu(body, variable=mode_var, values=list(labels.values()), font=parent.font).grid(row=5, column=1, sticky="ew")
    sv_var = ctk.BooleanVar()
    ctk.CTkCheckBox(body, text=text("sv"), variable=sv_var, font=parent.font).grid(row=6, column=0, columnspan=2, sticky="w", pady=8)
    ctk.CTkLabel(body, text=text("regions"), font=parent.font, justify="left", wraplength=650).grid(
        row=7, column=0, columnspan=2, sticky="w", pady=(14, 5))
    region_frame = ctk.CTkFrame(body, fg_color="transparent")
    region_frame.grid(row=8, column=0, columnspan=2, sticky="ew")
    region_frame.grid_columnconfigure((0, 1, 2), weight=1)
    for col, key in enumerate(("start", "end", "strength")):
        ctk.CTkLabel(region_frame, text=text(key), font=parent.font).grid(row=0, column=col, sticky="w")
    rows, retained = [], {}
    def add_region(region=None):
        if len(rows) >= 8:
            return
        region = region or {}
        row = []
        for col, key in enumerate(("start_s", "end_s", "strength")):
            var = ctk.StringVar(value=str(region.get(key, 1 if key == "strength" else "")))
            widget = ctk.CTkEntry(region_frame, textvariable=var, font=parent.font, width=120)
            widget.grid(row=len(rows)+1, column=col, sticky="ew", padx=(0, 8), pady=4)
            row.append((var, widget))
        rows.append((row, {k:region[k] for k in ("attack_s", "release_s") if k in region}))
    def populate(options):
        retained.clear()
        retained.update(options)
        for key, var in fields.items():
            var.set("" if options[key] is None else str(options[key]))
        mode_var.set(labels[options["highlight_mode"]])
        sv_var.set(options["highlight_sv"])
        for row, _ in rows:
            for _, widget in row:
                widget.destroy()
        rows.clear()
        for region in options["highlights"] or [{}]:
            add_region(region)
    def collect():
        options = dict(retained)
        for key, var in fields.items():
            value = var.get().strip()
            options[key] = int(value or 3) if key == "candidates" else float(value) if value else None
        options["highlight_mode"] = next(k for k, v in labels.items() if v == mode_var.get())
        options["highlight_sv"] = bool(sv_var.get())
        options["highlights"] = []
        if options["highlight_mode"] == "manual":
            for row, ramp in rows:
                a, b, strength = [v.get().strip() for v, _ in row]
                if a or b:
                    options["highlights"].append(dict(start_s=float(a), end_s=float(b), strength=float(strength or 1), **ramp))
        return validate_controls(options)
    def guarded(action):
        try:
            action()
        except (ValueError, OSError, UnicodeError) as exc:
            messagebox.showerror(text("title"), str(exc), parent=window)
    def apply():
        parent.generation_controls = collect()
        parent.refresh_control_summary()
        window.destroy()
    def load():
        path = filedialog.askopenfilename(parent=window, filetypes=[("JSON", "*.json")])
        if path:
            populate(load_control_file(path))
    def save():
        options = collect()
        path = filedialog.asksaveasfilename(parent=window, defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            if not path.lower().endswith(".json"):
                raise ValueError("Choose a .json file")
            atomic_json(Path(path), dict(schema="autoosu.control-plan/1", controls=options))
    ctk.CTkButton(body, text=text("add"), command=add_region, font=parent.font).grid(row=9, column=0, sticky="w", pady=8)
    ctk.CTkLabel(body, text=text("curve_hint"), font=parent.font, wraplength=650, justify="left").grid(
        row=10, column=0, columnspan=2, sticky="w", pady=8)
    buttons = ctk.CTkFrame(window, fg_color="transparent")
    buttons.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 16))
    for label, action in (("load", load), ("save", save), ("reset", lambda:populate(validate_controls())), ("apply", apply)):
        ctk.CTkButton(buttons, text=text(label), command=lambda f=action:guarded(f), width=140, font=parent.font).pack(side="left", padx=4)
    try:
        populate(validate_controls(parent.generation_controls))
    except ValueError:
        populate(validate_controls())
    # Also used by the real-widget integration check.
    window.control_form = dict(populate=populate, collect=collect, apply=apply)
    return window
