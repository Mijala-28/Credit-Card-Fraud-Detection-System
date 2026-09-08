import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_inspect(path="data/creditcard.csv"):
    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print()
    print("Nulls per column (total):", df.isnull().sum().sum())
    print()
    print("Duplicate rows:", df.duplicated().sum())
    print()
    print("Class balance:")
    print(df["Class"].value_counts())
    print(df["Class"].value_counts(normalize=True) * 100)

    return df

def clean_and_split(df, test_size=0.2, random_state=42):
    # Drop duplicates - same row twice would leak across train/test
    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"Dropped {before - df.shape[0]} duplicate rows")

    X = df.drop(columns=["Class"])
    y = df["Class"]

     # Stratified split so both sets keep the same fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"Train fraud rate: {y_train.mean()*100:.4f}%")
    print(f"Test fraud rate:  {y_test.mean()*100:.4f}%")

    # Fit scaler on TRAIN ONLY - test must stay unseen
    scaler = StandardScaler()
    X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    return X_train, X_test, y_train, y_test, scaler

def save_artifacts(X_train, X_test, y_train, y_test, scaler, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    X_train.to_csv(f"{out_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{out_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{out_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{out_dir}/y_test.csv", index=False)
    joblib.dump(scaler, f"{out_dir}/scaler.pkl")
    print(f"Saved train/test splits and scaler to {out_dir}/")



if __name__ == "__main__":
     df = load_and_inspect()
     X_train, X_test, y_train, y_test, scaler = clean_and_split(df)
     save_artifacts(X_train, X_test, y_train, y_test, scaler)
     
