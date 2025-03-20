# %%
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from pathlib import Path
import pandas as pd
from LoadSPA import Load_SPA


def spa_to_csv(filename):
    "Convert Thermo FTIR SPA to CSV files"
    try:
        path = Path(filename)
        output_path = path.with_suffix(".csv")

        df = Load_SPA(filename)
        df.to_csv(output_path)

    except Exception as e:
        # Handle the exception
        print(f"An error occurred: {e}")


def on_drop(event):
    filename = event.data
    if filename.startswith("{") and filename.endswith("}"):
        filename = filename[1:-1]
    spa_to_csv(filename)


root = TkinterDnD.Tk()
root.title("SPA to CSV Converter")

frame = tk.Frame(root, width=400, height=200, bg="lightgrey")
frame.pack_propagate(False)
frame.pack()

label = tk.Label(frame, text="Drag and drop a SPA file here", bg="lightgrey")
label.pack(expand=True)

frame.drop_target_register(DND_FILES)
frame.dnd_bind("<<Drop>>", on_drop)

root.mainloop()

# %%
