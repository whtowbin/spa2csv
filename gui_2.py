# %%
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from pathlib import Path
import pandas as pd


def csv_to_json(filename):
    "Convert CSV to JSON. Uses first row as array names"
    try:
        path = Path(filename)  # .name.split(".")[0]
        output_path = path.with_suffix(".json")

        df = pd.read_csv(filename)
        df.to_json(output_path)
        # df.to_json("test_output.json")
        # click.echo(f"{filename} converted to {output_path}")
    except Exception as e:
        # Handle the exception
        print(f"An error occurred: {e}")


def on_drop(event):
    filename = event.data
    if filename.startswith("{") and filename.endswith("}"):
        filename = filename[1:-1]
    csv_to_json(filename)


root = TkinterDnD.Tk()
root.title("CSV to JSON Converter")

frame = tk.Frame(root, width=400, height=200, bg="lightgrey")
frame.pack_propagate(False)
frame.pack()

label = tk.Label(frame, text="Drag and drop a CSV file here", bg="lightgrey")
label.pack(expand=True)

frame.drop_target_register(DND_FILES)
frame.dnd_bind("<<Drop>>", on_drop)

root.mainloop()

# %%
