from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, train_test_split

from data_cleaning import clean_used_car_data
from evaluation import calculate_regression_metrics, compare_models
from model_training import (
	MODEL_FEATURES,
	TARGET_COLUMN,
	apply_feature_engineering,
	build_model_pipelines,
	get_random_forest_param_grid,
)
from prediction import build_feature_row, parse_bool


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run full used-car pipeline: clean data, train model, and predict price."
	)
	parser.add_argument(
		"--raw-input",
		type=Path,
		default=Path("data/used_cars.csv"),
		help="Path to raw dataset.",
	)
	parser.add_argument(
		"--cleaned-output",
		type=Path,
		default=Path("data/cleaned_used_cars.csv"),
		help="Path for cleaned dataset output.",
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
	parser.add_argument(
		"--skip-predict",
		action="store_true",
		help="Skip final single-vehicle prediction step.",
	)
	parser.add_argument("--model-year", type=int, help="Vehicle model year for prediction.")
	parser.add_argument("--mileage", type=float, help="Vehicle mileage in miles.")
	parser.add_argument("--fuel-type", type=str, help="Fuel type (Gas, Diesel, Electric).")
	parser.add_argument("--brand", type=str, help="Brand name (for example: Toyota).")
	parser.add_argument(
		"--accident-reported",
		type=str,
		help="Whether an accident was reported (true/false).",
	)
	parser.add_argument(
		"--clean-title-flag",
		type=str,
		help="Whether the title is clean (true/false).",
	)
	return parser.parse_args()


def prompt_prediction_inputs(args: argparse.Namespace) -> argparse.Namespace:
	if args.model_year is None:
		args.model_year = int(input("Model year: ").strip())
	if args.mileage is None:
		args.mileage = float(input("Mileage (miles): ").strip())
	if not args.fuel_type:
		args.fuel_type = input("Fuel type (Gas/Diesel/Electric): ").strip()
	if not args.brand:
		args.brand = input("Brand: ").strip()
	if args.accident_reported is None:
		args.accident_reported = input("Accident reported? (true/false): ").strip()
	if args.clean_title_flag is None:
		args.clean_title_flag = input("Clean title? (true/false): ").strip()
	return args


def run_cleaning(raw_input: Path, cleaned_output: Path) -> pd.DataFrame:
	if not raw_input.exists():
		raise FileNotFoundError(f"Input file not found: {raw_input}")

	raw_df = pd.read_csv(raw_input)
	cleaned_df = clean_used_car_data(raw_df)

	cleaned_output.parent.mkdir(parents=True, exist_ok=True)
	cleaned_df.to_csv(cleaned_output, index=False)

	print(f"Raw shape: {raw_df.shape}")
	print(f"Cleaned shape: {cleaned_df.shape}")
	print(f"Saved cleaned dataset to: {cleaned_output.resolve()}")
	return cleaned_df


def run_training(
	cleaned_df: pd.DataFrame,
	model_output: Path,
	metrics_output: Path,
):
	feat = apply_feature_engineering(cleaned_df)
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
	trained_pipelines = {}

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

	model_output.parent.mkdir(parents=True, exist_ok=True)
	metrics_output.parent.mkdir(parents=True, exist_ok=True)

	# Export the best-performing model for downstream prediction.
	dump(trained_pipelines[best_model_name], model_output)
	metrics_df.to_csv(metrics_output, index=False)

	print(f"Training rows: {len(X_train)}")
	print(f"Test rows: {len(X_test)}")
	print("Model comparison:")
	print(metrics_df.to_string(index=False))
	print(f"Selected best model: {best_model_name}")
	print(f"Saved model to: {model_output.resolve()}")
	print(f"Saved metrics to: {metrics_output.resolve()}")

	return trained_pipelines[best_model_name]


def run_prediction(model, args: argparse.Namespace) -> None:
	if args.skip_predict:
		print("Skipped prediction step (--skip-predict).")
		return

	args = prompt_prediction_inputs(args)
	feature_row = build_feature_row(
		model_year=args.model_year,
		mileage_mi=args.mileage,
		fuel_type=args.fuel_type,
		brand=args.brand,
		accident_reported=parse_bool(args.accident_reported),
		clean_title_flag=parse_bool(args.clean_title_flag),
	)

	predicted_price = model.predict(feature_row)[0]

	print("Input features:")
	print(feature_row.to_string(index=False))
	print(f"\nPredicted used-car price: ${predicted_price:,.2f}")


def main() -> None:
	args = parse_args()
	cleaned_df = run_cleaning(args.raw_input, args.cleaned_output)
	model = run_training(cleaned_df, args.model_output, args.metrics_output)
	run_prediction(model, args)


if __name__ == "__main__":
	main()