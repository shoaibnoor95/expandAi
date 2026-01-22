import pandas as pd
import os

def check_integrity(submission_file, test_file):
    print(f"Checking {submission_file} against {test_file}...")
    
    try:
        sub_df = pd.read_csv(submission_file)
        test_df = pd.read_csv(test_file)
        
        print(f"Submission rows: {len(sub_df)}")
        print(f"Test rows: {len(test_df)}")
        
        if len(sub_df) != len(test_df):
            print(f"WARNING: Row count mismatch! Missing {len(test_df) - len(sub_df)} rows.")
            
        # Check for nulls/NaNs
        if sub_df.isnull().values.any():
            print("FOUND NULLS via isnull()")
            print(sub_df[sub_df.isnull().any(axis=1)].head())
        else:
            print("No nulls found via isnull()")
            
        # Check for string "nan" or "None" or empty strings in the first label column (e.g. 'VA')
        # Assuming first column is Filename
        if 'VA' in sub_df.columns:
            print("Checking for string 'nan' or empty values...")
            # Check unique values in a label column just to see if anything looks weird
            print(f"Unique values in 'VA' (head 10): {sub_df['VA'].unique()[:10]}")
            
        # Check if Filenames match
        if 'Filename' in sub_df.columns and 'Filename' in test_df.columns:
            sub_files = set(sub_df['Filename'])
            test_files = set(test_df['Filename'])
            
            missing_in_sub = test_files - sub_files
            missing_in_test = sub_files - test_files
            
            if missing_in_sub:
                print(f"WARNING: {len(missing_in_sub)} filenames from test.csv are missing in submission.csv")
            if missing_in_test:
                print(f"WARNING: {len(missing_in_test)} filenames in submission.csv are NOT in test.csv")
                
    except Exception as e:
        print(f"Error analysis: {e}")

if __name__ == "__main__":
    check_integrity("submission.csv", "test.csv")
