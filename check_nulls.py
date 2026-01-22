import pandas as pd

def check_nulls(csv_file):
    try:
        print(f"Reading {csv_file}...")
        df = pd.read_csv(csv_file)
        
        if df.isnull().values.any():
            print("Found null values!")
            null_counts = df.isnull().sum()
            print("Null counts per column:")
            print(null_counts[null_counts > 0])
            
            # Show a few examples
            print("\nRows with nulls:")
            print(df[df.isnull().any(axis=1)].head())
        else:
            print("No null values found.")
            
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

if __name__ == "__main__":
    check_nulls("submission.csv")
