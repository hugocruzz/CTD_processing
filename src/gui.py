import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path

# Import your processing function
from main import process_all_files


def browse_folder():
    folder = filedialog.askdirectory()
    if folder:
        source_var.set(folder)


def run_processing():

    data_dir = source_var.get()

    if not data_dir:
        messagebox.showerror("Error", "Please select a source folder.")
        return

    sort_by = sort_var.get()

    campaign = Path(data_dir).name

    Level1_output = os.path.join("data", "Level1", campaign)
    Level2_output = os.path.join("data", "Level2", campaign)
    Level2B_output = os.path.join("data", "Level2B", campaign)

    processing_mode = None
    split_profile = False
    sort_by_pressure = False

    status_var.set("Processing...")
    root.update()

    try:

        process_all_files(
            data_dir,
            Level1_output,
            Level2_output,
            Level2B_output,
            processing_mode,
            split_profile,
            sort_by,
            sort_by_pressure,
        )

        status_var.set(
            "Finished.\nData output for A2PS software in data/Level2B"
        )

        messagebox.showinfo(
            "Finished",
            "Data output for A2PS software in data/Level2B",
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))
        status_var.set("Failed")


root = tk.Tk()
root.title("SubOcean Profile Processing")
root.geometry("600x180")

source_var = tk.StringVar()
sort_var = tk.StringVar(value="datetime")
status_var = tk.StringVar(value="Ready")

# Source folder
ttk.Label(root, text="Source folder").grid(
    row=0, column=0, padx=10, pady=10, sticky="w"
)

ttk.Entry(
    root,
    textvariable=source_var,
    width=60,
).grid(
    row=0,
    column=1,
    padx=5,
    pady=10,
)

ttk.Button(
    root,
    text="Browse...",
    command=browse_folder,
).grid(
    row=0,
    column=2,
    padx=10,
)

# Sort option
ttk.Label(root, text="Sort by").grid(
    row=1,
    column=0,
    padx=10,
    pady=10,
    sticky="w",
)

ttk.Combobox(
    root,
    textvariable=sort_var,
    values=["datetime", "depth"],
    state="readonly",
    width=15,
).grid(
    row=1,
    column=1,
    sticky="w",
)

# Run button
ttk.Button(
    root,
    text="Run",
    command=run_processing,
).grid(
    row=2,
    column=1,
    pady=20,
)

# Status
ttk.Label(
    root,
    textvariable=status_var,
).grid(
    row=3,
    column=0,
    columnspan=3,
    pady=10,
)

root.mainloop()