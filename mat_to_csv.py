import scipy.io
import h5py
import numpy as np
import pandas as pd
import os

def mat_to_csv(mat_path, output_folder="."):
    base_name = os.path.splitext(os.path.basename(mat_path))[0]

    try:
        data = scipy.io.loadmat(mat_path)
        print("Loaded with scipy.io.loadmat")
        print("\nKeys in file:", list(data.keys()))
        print("\nType of 'data':", type(data.get("data", None)))
        print("\nStructure of 'data':\n", data.get("data", None))

        # remove MATLAB internal keys
        data = {k: v for k, v in data.items() if not k.startswith("__")}

        for key, value in data.items():
            arr = np.array(value)

            # only process numeric 2D arrays
            if not np.issubdtype(arr.dtype, np.number) or arr.ndim < 2:
                print(f"⚠️  Skipping variable '{key}' (not a numeric 2D array)")
                continue

            df = pd.DataFrame(arr)
            out_path = os.path.join(output_folder, f"{base_name}_{key}.csv")
            df.to_csv(out_path, index=False)
            print(f"✅ Saved: {out_path}")

    except NotImplementedError:
        print("File is MATLAB v7.3 format, using h5py...")
        with h5py.File(mat_path, "r") as f:
            for key in f.keys():
                arr = np.array(f[key])
                if arr.ndim > 1:
                    arr = arr.T
                if not np.issubdtype(arr.dtype, np.number):
                    print(f"⚠️  Skipping '{key}' (non-numeric data)")
                    continue
                df = pd.DataFrame(arr)
                out_path = os.path.join(output_folder, f"{base_name}_{key}.csv")
                df.to_csv(out_path, index=False)
                print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    mat_file = r"C:\Users\KIIT\Desktop\major_project _2\A07.mat" 
    mat_to_csv(mat_file)
