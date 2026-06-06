import pandas as pd

train_path = r'C:\Users\khush\Downloads\student_resource\dataset\train.csv'
df = pd.read_csv(train_path, nrows=50000)

print("Columns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nTarget Distribution:")
if 'scored_flag' in df.columns:
    print(df['scored_flag'].value_counts(normalize=True))
else:
    print("No 'scored_flag' found.")
