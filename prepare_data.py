from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_PATH = DATA_DIR / "heart_disease.csv"


def load_heart_disease_dataset() -> pd.DataFrame:
    """Carrega o dataset Cleveland Heart Disease da UCI."""
    heart_disease = fetch_ucirepo(id=45)
    features = heart_disease.data.features
    target = heart_disease.data.targets
    return features.join(target)


def build_binary_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    """Cria um rótulo binário: 1 para presença de doença e 0 para ausência.

    A documentação oficial da UCI indica que a variável num representa doença
    angiográfica, com valores 0 para ausência e 1-4 para presença de doença.
    Não há justificativa para interpretar 1,2,3,4 como severidade clínica em
    ordem crescente.
    """
    result = df.copy()
    result["diagnostico_binario"] = (result["num"] > 0).astype(int)
    return result


def save_dataset(df: pd.DataFrame, path: Path = DATA_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    df = load_heart_disease_dataset()
    save_dataset(df)
    print(f"Dataset salvo em {DATA_PATH}")
    print(df.shape)