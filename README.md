# spa2csv

A small drag-and-drop window that converts Thermo FTIR `.spa` files to CSV. Drop a `.spa` file on the window and a CSV with the same name is written next to it, with a row index, a wavenumber column (`cm^-1`) and an absorbance column named after the sample.

```bash
uv run python gui.py
```

To read a spectrum in your own code, `LoadSPA.Load_SPA(path)` returns it as a pandas DataFrame.

`gui_2.py` is the same window converting CSV files to JSON. The GitHub Actions workflow builds standalone apps of `gui.py` for macOS, Linux and Windows with Nuitka, installing from `requirements.txt`.

## License

The code is under the MIT License (see LICENSE).
