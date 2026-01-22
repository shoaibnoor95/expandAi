import pandas as pd
try:
    sub = pd.read_csv("submission.csv")
    test = pd.read_csv("test.csv")
    print(f"Submission Rows: {len(sub)}")
    print(f"Test Rows: {len(test)}")
except Exception as e:
    print(f"Error: {e}")
