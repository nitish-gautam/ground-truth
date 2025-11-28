#!/usr/bin/env python3
"""
Script to explore HS2 Excel file structure
"""
import pandas as pd
import sys

file_path = sys.argv[1] if len(sys.argv) > 1 else "/datasets/hs2/rawdata/hs2_monthly_monitoring_data_December_2024/hs2_noise_data_areanorth_birmingham_december_2024.xlsx"

print(f"Reading: {file_path}\n")

# Read Excel file
xl = pd.ExcelFile(file_path)

print(f"Sheet names: {xl.sheet_names}\n")

# Read Metadata sheet
print("=" * 80)
print("METADATA SHEET:")
print("=" * 80)
df_meta = pd.read_excel(file_path, sheet_name='Metadata')
print(df_meta.head(10))

# Read second sheet (first actual data sheet) with proper header row
print("\n" + "=" * 80)
print(f"DATA SHEET: {xl.sheet_names[1]}")
print("=" * 80)

# Try reading with header at row 2 (skip first 2 rows)
df = pd.read_excel(file_path, sheet_name=1, header=2)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")
print("First 10 rows:")
print(df.head(10))
print("\nData types:")
print(df.dtypes)
print("\nRow 0 actual:")
print(df.iloc[0])
print("\nNon-null counts:")
print(df.count())
