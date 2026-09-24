"""Non-blocking source-check window, independent of generation and GPU workers."""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

from .i18n import language, tr
from .provenance import atomic_json
from .source_check import check_source, format_report


def open_source_window(parent):
    import customtkinter as ctk

    window = ctk.CTkToplevel(parent)
    window.title(tr("run.check_source"))
    window.geometry("850x580")
    window.minsize(720, 440)
    window.grid_columnconfigure(0, weight=1)
    window.grid_rowconfigure(2, weight=1)
    # Capture language on opening so an in-progress report keeps consistent wording.
    lang = language()
    state = {"report": None, "busy": False, "after_id": None}
    messages = queue.Queue()
    explanation = ctk.CTkLabel(window, text=tr("source.explanation"), font=parent.font,
                                justify="left", anchor="w", wraplength=790)
    explanation.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 12))
    buttons = ctk.CTkFrame(window, fg_color="transparent")
    buttons.grid(row=1, column=0, sticky="ew", padx=20)
    recursive = ctk.BooleanVar(value=False)
    text_box = ctk.CTkTextbox(window, font=parent.font, wrap="word")
    text_box.grid(row=2, column=0, sticky="nsew", padx=20, pady=14)
    text_box.configure(state="disabled")

    def display(text):
        text_box.configure(state="normal")
        text_box.delete("1.0", "end")
        text_box.insert("1.0", text)
        text_box.configure(state="disabled")

    def start(path):
        if not path or state["busy"]:
            return
        state.update(busy=True, report=None)
        for b in (file_button, folder_button, save_button):
            b.configure(state="disabled")
        display(tr("source.checking"))
        include_subdirs = recursive.get()
        def work():
            try:
                messages.put((check_source(path, recursive=include_subdirs), None))
            except Exception as exc:
                messages.put((None, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def save():
        if state["report"] is None:
            return
        path = filedialog.asksaveasfilename(parent=window, defaultextension=".json", initialfile="source-report.json",
                                            filetypes=[("JSON", "*.json")])
        if path:
            try:
                if Path(path).suffix.lower() != ".json":
                    raise ValueError("Choose a .json report filename")
                atomic_json(Path(path), state["report"])
            except (OSError, ValueError) as exc:
                messagebox.showerror("AUTO-OSU", str(exc), parent=window)

    file_button = ctk.CTkButton(buttons, text=tr("source.files"), font=parent.font,
                                command=lambda: start(filedialog.askopenfilename(parent=window, filetypes=[("osu!", "*.osu *.osz")])))
    file_button.pack(side="left")
    folder_button = ctk.CTkButton(buttons, text=tr("source.folder"), font=parent.font,
                                  command=lambda: start(filedialog.askdirectory(parent=window)))
    folder_button.pack(side="left", padx=8)
    ctk.CTkCheckBox(buttons, text=tr("source.recursive"), variable=recursive, font=parent.font).pack(side="left", padx=8)
    save_button = ctk.CTkButton(window, text=tr("source.save"), font=parent.font, command=save, state="disabled")
    save_button.grid(row=3, column=0, sticky="e", padx=20, pady=(0, 16))

    def poll():
        try:
            report, error = messages.get_nowait()
            state.update(busy=False, report=report)
            display(error or format_report(report, lang))
            file_button.configure(state="normal")
            folder_button.configure(state="normal")
            save_button.configure(state="normal" if report else "disabled")
        except queue.Empty:
            pass
        state["after_id"] = window.after(100, poll)

    def close():
        if state["after_id"] is not None:
            window.after_cancel(state["after_id"])
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", close)
    state["after_id"] = window.after(100, poll)
    # Useful to embedding callers and UI validation, without touching the generator.
    window.check_path = start
    window.source_report = state
    window.after(50, window.lift)
    return window
