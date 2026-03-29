from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(
	y_true: pd.Series | list[float],
	y_pred: pd.Series | list[float],
	model_name: str = "model",
) -> dict[str, float | int | str]:
	"""Return standard regression metrics for a single model."""
	y_true_series = pd.Series(y_true)
	y_pred_series = pd.Series(y_pred)

	mae = mean_absolute_error(y_true_series, y_pred_series)
	rmse = mean_squared_error(y_true_series, y_pred_series, squared=False)
	r2 = r2_score(y_true_series, y_pred_series)

	return {
		"model": model_name,
		"mae": round(mae, 4),
		"rmse": round(rmse, 4),
		"r2": round(r2, 4),
		"n_samples": int(len(y_true_series)),
	}


def compare_models(metrics: list[dict[str, float | int | str]]) -> pd.DataFrame:
	"""Create a sorted metrics table for model comparison."""
	if not metrics:
		return pd.DataFrame(columns=["model", "mae", "rmse", "r2", "n_samples"])

	comparison = pd.DataFrame(metrics)
	return comparison.sort_values(by=["rmse", "mae"], ascending=True).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate prediction quality from a CSV containing true and predicted values."
	)
	parser.add_argument(
		"--input",
		type=Path,
		required=True,
		help="Path to CSV containing columns for y_true and y_pred.",
	)
	parser.add_argument(
		"--true-col",
		type=str,
		default="y_true",
		help="Column name for true values.",
	)
	parser.add_argument(
		"--pred-col",
		type=str,
		default="y_pred",
		help="Column name for predicted values.",
	)
	parser.add_argument(
		"--model-name",
		type=str,
		default="model",
		help="Label used for reporting this model's metrics.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("results/evaluation.csv"),
		help="Path to save computed metrics CSV.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if not args.input.exists():
		raise FileNotFoundError(f"Input file not found: {args.input}")

	df = pd.read_csv(args.input)
	missing = [c for c in [args.true_col, args.pred_col] if c not in df.columns]
	if missing:
		raise KeyError(f"Missing required columns in input data: {missing}")

	metrics = calculate_regression_metrics(
		y_true=df[args.true_col],
		y_pred=df[args.pred_col],
		model_name=args.model_name,
	)

	metrics_df = compare_models([metrics])
	args.output.parent.mkdir(parents=True, exist_ok=True)
	metrics_df.to_csv(args.output, index=False)

	print(metrics_df.to_string(index=False))
	print(f"Saved evaluation to: {args.output.resolve()}")


if __name__ == "__main__":
	main()