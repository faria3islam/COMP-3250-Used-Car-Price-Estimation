from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from evaluation import calculate_regression_metrics, compare_models


CURRENT_YEAR = pd.Timestamp.now().year
MODEL_FEATURES = [
	"car_age",
	"scaled_mileage",
	"fuel_type_encoded",
	"brand",
	"accident_reported",
	"clean_title_flag",
]
TARGET_COLUMN = "price_usd"


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
	"""Create engineered features used by the training pipeline."""
	feat = df.copy()

	feat["model_year_numeric"] = pd.to_numeric(feat.get("model_year"), errors="coerce")
	feat["car_age"] = CURRENT_YEAR - feat["model_year_numeric"]
	feat.loc[feat["car_age"] < 0, "car_age"] = np.nan

	if "mileage_mi" in feat.columns:
		mileage_base = pd.to_numeric(feat["mileage_mi"], errors="coerce")
	else:
		mileage_base = pd.to_numeric(
			feat.get("milage", pd.Series(index=feat.index, dtype="object"))
			.astype(str)
			.str.replace(r"[^0-9]", "", regex=True),
			errors="coerce",
		)
	feat["scaled_mileage"] = mileage_base / 1000

	fuel_map = {"gas": 0, "diesel": 1, "electric": 2}
	fuel_normalized = feat.get("fuel_type", pd.Series(index=feat.index, dtype="object"))
	fuel_normalized = fuel_normalized.astype(str).str.strip().str.lower()
	fuel_normalized = fuel_normalized.replace({"nan": np.nan, "none": np.nan})
	fuel_normalized = fuel_normalized.replace(
		{
			"gasoline": "gas",
			"battery electric": "electric",
			"electric motor": "electric",
		}
	)
	feat["fuel_type_encoded"] = fuel_normalized.map(fuel_map)

	if TARGET_COLUMN not in feat.columns and "price" in feat.columns:
		feat[TARGET_COLUMN] = pd.to_numeric(
			feat["price"].astype(str).str.replace(r"[\$,]", "", regex=True),
			errors="coerce",
		)

	return feat


def build_preprocessor() -> ColumnTransformer:
	numeric_features = ["car_age", "scaled_mileage", "fuel_type_encoded"]
	categorical_features = ["brand", "accident_reported", "clean_title_flag"]

	return ColumnTransformer(
		transformers=[
			(
				"num",
				Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
				numeric_features,
			),
			(
				"cat",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="most_frequent")),
						("onehot", OneHotEncoder(handle_unknown="ignore")),
					]
				),
				categorical_features,
			),
		]
	)


def build_model_pipelines() -> dict[str, Pipeline]:
	preprocessor = build_preprocessor()

	baseline_linear = Pipeline(
		steps=[
			("preprocess", preprocessor),
			("model", LinearRegression()),
		]
	)

	advanced_random_forest = Pipeline(
		steps=[
			("preprocess", preprocessor),
			(
				"model",
				RandomForestRegressor(
					n_estimators=300,
					random_state=42,
					n_jobs=-1,
				),
			),
		]
	)

	return {
		"linear_regression": baseline_linear,
		"random_forest": advanced_random_forest,
	}


def get_random_forest_param_grid() -> list[dict[str, int | None]]:
	"""Small, explicit grid for advanced-model tuning."""
	return [
		{"n_estimators": 200, "max_depth": 10},
		{"n_estimators": 300, "max_depth": 20},
		{"n_estimators": 500, "max_depth": None},
	]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train used-car price model from cleaned dataset."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("data/cleaned_used_cars.csv"),
		help="Path to cleaned input CSV.",
	)
	parser.add_argument(
		"--model-output",
		type=Path,
		default=Path("results/price_model.joblib"),
		help="Path to save trained model artifact.",
	)
	parser.add_argument(
		"--metrics-output",
		type=Path,
		default=Path("results/evaluation.csv"),
		help="Path to save evaluation metrics CSV.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if not args.input.exists():
		raise FileNotFoundError(f"Input file not found: {args.input}")

	df = pd.read_csv(args.input)
	feat = apply_feature_engineering(df)

	required = MODEL_FEATURES + [TARGET_COLUMN]
	missing_columns = [c for c in required if c not in feat.columns]
	if missing_columns:
		raise KeyError(f"Missing required columns for training: {missing_columns}")

	train_df = feat[required].dropna(subset=[TARGET_COLUMN]).copy()

	X = train_df[MODEL_FEATURES]
	y = train_df[TARGET_COLUMN]

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	pipelines = build_model_pipelines()
	all_metrics: list[dict[str, float | int | str]] = []
	trained_pipelines: dict[str, Pipeline] = {}

	linear_pipeline = pipelines["linear_regression"]
	linear_pipeline.fit(X_train, y_train)
	linear_preds = linear_pipeline.predict(X_test)
	linear_metrics = calculate_regression_metrics(
		y_true=y_test,
		y_pred=linear_preds,
		model_name="linear_regression",
	)
	all_metrics.append(
		{
			**linear_metrics,
			"n_train": len(X_train),
			"n_test": len(X_test),
		}
	)
	trained_pipelines["linear_regression"] = linear_pipeline

	rf_base = pipelines["random_forest"]
	for params in ParameterGrid(get_random_forest_param_grid()):
		rf_pipeline = clone(rf_base)
		rf_pipeline.set_params(
			model__n_estimators=params["n_estimators"],
			model__max_depth=params["max_depth"],
		)
		rf_pipeline.fit(X_train, y_train)
		rf_preds = rf_pipeline.predict(X_test)
		model_label = (
			"random_forest"
			f"[n_estimators={params['n_estimators']},max_depth={params['max_depth']}]"
		)
		rf_metrics = calculate_regression_metrics(
			y_true=y_test,
			y_pred=rf_preds,
			model_name=model_label,
		)
		all_metrics.append(
			{
				**rf_metrics,
				"n_train": len(X_train),
				"n_test": len(X_test),
			}
		)
		trained_pipelines[model_label] = rf_pipeline

	metrics_df = compare_models(all_metrics)
	best_model_name = str(metrics_df.iloc[0]["model"])

	args.model_output.parent.mkdir(parents=True, exist_ok=True)
	args.metrics_output.parent.mkdir(parents=True, exist_ok=True)

	# Export the best-performing model based on comparison metrics.
	dump(trained_pipelines[best_model_name], args.model_output)
	metrics_df.to_csv(args.metrics_output, index=False)

	print(f"Training rows: {len(X_train)}")
	print(f"Test rows: {len(X_test)}")
	print("Model comparison:")
	print(metrics_df.to_string(index=False))
	print(f"Selected best model: {best_model_name}")
	print(f"Saved model to: {args.model_output.resolve()}")
	print(f"Saved metrics to: {args.metrics_output.resolve()}")


if __name__ == "__main__":
	main()
