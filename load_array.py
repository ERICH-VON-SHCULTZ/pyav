#!/usr/bin/env python3
"""
load_array.py

Utility to load .npy or .npz files and display their shape, dtype, and type.
Optionally preview the first N elements.

Usage:
    python load_array.py myfile.npy
    python load_array.py myfile.npz --head 10
    python load_array.py --test
"""

import os
import sys
import argparse
import numpy as np


def load_to_array(filename):
    """
    Load a file that may be either a .npy or .npz file,
    and always return a NumPy array.

    If it's an .npz file with multiple arrays, they will be stacked
    if possible, otherwise returned as an object array.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext not in (".npy", ".npz"):
        raise ValueError(f"Unsupported file type '{ext}'. Only .npy and .npz are allowed.")

    try:
        data = np.load(filename, allow_pickle=True)

        # Case: npz file
        if isinstance(data, np.lib.npyio.NpzFile):
            arrays = [data[key] for key in data.files]
            data.close()
            if len(arrays) == 1:
                return arrays[0]
            else:
                try:
                    return np.stack(arrays)
                except ValueError:
                    return np.array(arrays, dtype=object)

        # Case: npy file
        return data

    except Exception as e:
        raise ValueError(f"Error loading '{filename}': {e}")


def run_test():
    """Generate small .npy and .npz files and test loading them."""
    npy_file = "test_array.npy"
    npz_file = "test_arrays.npz"

    arr1 = np.arange(12).reshape(3, 4)
    arr2 = np.linspace(0, 1, 5)

    np.save(npy_file, arr1)
    np.savez(npz_file, first=arr1, second=arr2)

    print("🧪 Running test harness...")
    for f in [npy_file, npz_file]:
        try:
            arr = load_to_array(f)
            print(f"✅ Loaded '{f}' successfully.")
            print(f"   Shape: {arr.shape}")
            print(f"   Dtype: {arr.dtype}")
            print(f"   Preview: {arr.ravel()[:5]} ...")
        except Exception as e:
            print(f"❌ Failed to load '{f}': {e}")

    print("🧹 Cleaning up test files...")
    for f in [npy_file, npz_file]:
        if os.path.exists(f):
            os.remove(f)


def main():
    parser = argparse.ArgumentParser(
        description="Load a .npy or .npz file and show basic info."
    )
    parser.add_argument("filename", nargs="?", help="Path to the .npy or .npz file")
    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="Preview the first N elements of the array (flattened view)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a test harness (creates and loads example .npy/.npz files)."
    )

    args = parser.parse_args()

    if args.test:
        run_test()
        return

    if not args.filename:
        print("Usage: python load_array.py <filename.npy|filename.npz> [--head N]")
        sys.exit(1)

    try:
        arr = load_to_array(args.filename)
        print(f"✅ Loaded '{args.filename}' successfully.")
        print(f"   Shape: {arr.shape}")
        print(f"   Dtype: {arr.dtype}")
        print(f"   Type:  {type(arr)}")

        if args.head > 0:
            flat = arr.ravel()
            n = min(args.head, flat.size)
            print(f"   First {n} element(s): {flat[:n]}")

    except Exception as e:
        print(f"❌ Failed to load '{args.filename}': {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
