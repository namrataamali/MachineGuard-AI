from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import requests
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
URL = 'https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip'

def train():
    response = requests.get(URL, timeout=90)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as z:
        csv_name = next(name for name in z.namelist() if name.lower().endswith('.csv'))
        with z.open(csv_name) as f:
            raw = pd.read_csv(f)
    df = raw.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
    df = pd.get_dummies(df, columns=['Type'], dtype=int)
    X = df.drop(columns='Machine failure')
    y = df['Machine failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, BASE / 'machineguard_model.pkl')
    joblib.dump(X.columns.tolist(), BASE / 'feature_names.pkl')
    return model

if __name__ == '__main__':
    train()
    print('Model trained and saved.')
