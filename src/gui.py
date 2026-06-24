import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path

# Import your processing function
from main import process_all_files


def browse_input_folder():
    folder = filedialog.askdirectory()

    if folder:
        source_var.set(folder)

        # Default output folder = input folder
        if not output_var.get():
            output_var.set(folder)


def browse_output_folder():
    folder = filedialog.askdirectory()

    if folder:
        output_var.set(folder)


def run_processing():

    data_dir = source_var.get()

    if not data_dir:
        messagebox.showerror("Error", "Please select a source folder.")
        return

    output_root = output_var.get()

    if not output_root:
        output_root = data_dir

    sort_by = sort_var.get()

    campaign = Path(data_dir).name

    Level1_output = os.path.join(
        output_root,
        "data",
        "Level1",
        campaign,
    )

    Level2_output = os.path.join(
        output_root,
        "data",
        "Level2",
        campaign,
    )

    Level2B_output = os.path.join(
        output_root,
        "data",
        "Level2B",
        campaign,
    )

    processing_mode = None
    split_profile = False
    sort_by_pressure = True

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
            f"Finished.\nData output for A2PS software in:\n{Level2B_output}"
        )

        messagebox.showinfo(
            "Finished",
            f"Data output for A2PS software in:\n\n{Level2B_output}",
        )

    except Exception as e:

        messagebox.showerror("Error", str(e))
        status_var.set("Failed")


root = tk.Tk()
root.title("SubOcean Profile Processing")
root.geometry("750x250")

source_var = tk.StringVar()
output_var = tk.StringVar()
sort_var = tk.StringVar(value="datetime")
status_var = tk.StringVar(value="Ready")

# =========================
# Input folder
# =========================

ttk.Label(
    root,
    text="Source folder",
).grid(
    row=0,
    column=0,
    padx=10,
    pady=10,
    sticky="w",
)

ttk.Entry(
    root,
    textvariable=source_var,
    width=75,
).grid(
    row=0,
    column=1,
    padx=5,
    pady=10,
)

ttk.Button(
    root,
    text="Browse...",
    command=browse_input_folder,
).grid(
    row=0,
    column=2,
    padx=10,
)

# =========================
# Output folder
# =========================

ttk.Label(
    root,
    text="Output folder",
).grid(
    row=1,
    column=0,
    padx=10,
    pady=10,
    sticky="w",
)

ttk.Entry(
    root,
    textvariable=output_var,
    width=75,
).grid(
    row=1,
    column=1,
    padx=5,
    pady=10,
)

ttk.Button(
    root,
    text="Browse...",
    command=browse_output_folder,
).grid(
    row=1,
    column=2,
    padx=10,
)

# =========================
# Sort option
# =========================

ttk.Label(
    root,
    text="Sort by",
).grid(
    row=2,
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
    row=2,
    column=1,
    sticky="w",
)

# =========================
# Run button
# =========================

ttk.Button(
    root,
    text="Run",
    command=run_processing,
).grid(
    row=3,
    column=1,
    pady=20,
)

# =========================
# Status
# =========================

ttk.Label(
    root,
    textvariable=status_var,
).grid(
    row=4,
    column=0,
    columnspan=3,
    pady=10,
)

root.mainloop()