import pandas as pd

from prepare_data import build_binary_diagnosis, load_heart_disease_dataset


def test_build_binary_diagnosis_is_presence_absence():
    df = pd.DataFrame({"num": [0, 1, 2, 3, 4]})

    result = build_binary_diagnosis(df)

    assert list(result["diagnostico_binario"].unique()) == [0, 1]
    assert result.loc[result["num"] == 0, "diagnostico_binario"].all() == 0
    assert result.loc[result["num"] > 0, "diagnostico_binario"].all() == 1


def test_load_heart_disease_dataset_keeps_uci_structure():
    df = load_heart_disease_dataset()

    assert df.shape[0] == 303
    assert set(df.columns) == {
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "num",
    }
    assert df["num"].between(0, 4).all()
