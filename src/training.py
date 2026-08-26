from __future__ import annotations

from math import sqrt
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import MODEL_PATH
from .data import dataset_summary, load_dataset
from .features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    historical_profiles,
    training_features,
)


def _pipeline(regressor) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numericas", "passthrough", NUMERIC_FEATURES),
        ]
    )
    return Pipeline([("preprocesado", preprocessor), ("modelo", regressor)])


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_save(model_path: Path = MODEL_PATH, verbose: bool = True) -> dict:
    df, dataset_path = load_dataset()
    summary = dataset_summary(df)
    expected_friends = 8
    if len(summary["amigos"]) != expected_friends:
        raise ValueError(
            f"Se esperaban {expected_friends} amigos y hay {len(summary['amigos'])}: "
            f"{summary['amigos']}"
        )

    if verbose:
        print(f"Cargando dataset: {dataset_path.name}")
        print(f"{summary['observaciones']:,} observaciones".replace(",", "."))
        print(f"{summary['quedadas']} quedadas")
        print(f"{len(summary['amigos'])} amigos: {', '.join(summary['amigos'])}")
        print(f"Valores nulos: {summary['nulos']}")
        delay = summary["retraso"]
        print(
            "Retraso (min): "
            f"media {delay['media']:.1f}, mediana {delay['mediana']:.1f}, "
            f"rango {delay['min']:.0f} a {delay['max']:.0f}"
        )

    X = training_features(df)
    y = df["retraso_min"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=df["quedada_id"]))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # El baseline aprende solo del train para no filtrar información del test.
    friend_means = df.iloc[train_idx].groupby("amigo")["retraso_min"].mean()
    global_mean = float(y_train.mean())
    baseline_pred = df.iloc[test_idx]["amigo"].map(friend_means).fillna(global_mean)
    results: dict[str, dict[str, float]] = {
        "Baseline por amigo": _metrics(y_test, baseline_pred)
    }

    candidates = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }
    fitted: dict[str, Pipeline] = {}
    if verbose:
        print("\nEntrenando modelos...")
    for name, regressor in candidates.items():
        pipeline = _pipeline(regressor)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        results[name] = _metrics(y_test, pipeline.predict(X_test))

    selected_name = min(candidates, key=lambda name: results[name]["mae"])
    artifact = {
        "pipeline": fitted[selected_name],
        "model_name": selected_name,
        "metrics": results,
        "profiles": historical_profiles(df),
        "friends": df["amigo"].drop_duplicates().tolist(),
        "plan_types": sorted(df["tipo_plan"].unique().tolist()),
        "dataset_path": str(dataset_path),
        "train_groups": sorted(df.iloc[train_idx]["quedada_id"].unique().tolist()),
        "test_groups": sorted(df.iloc[test_idx]["quedada_id"].unique().tolist()),
        "feature_columns": X.columns.tolist(),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    if verbose:
        for name, metrics in results.items():
            print(
                f"{name:20} MAE {metrics['mae']:.2f} min | "
                f"RMSE {metrics['rmse']:.2f} | R² {metrics['r2']:.3f}"
            )
        print(f"\n✓ Seleccionado: {selected_name}")
        print(f"✓ Modelo guardado en: {model_path.relative_to(model_path.parents[1])}")
    return artifact

