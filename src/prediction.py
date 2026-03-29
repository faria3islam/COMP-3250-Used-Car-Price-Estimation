from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load


CURRENT_YEAR = pd.Timestamp.now().year


def parse_bool(value: str) -> bool:
	normalized = value.strip().lower()
	if normalized in {"true", "t", "yes", "y", "1"}:
		return True
	if normalized in {"false", "f", "no", "n", "0"}:
		return False
	raise ValueError(f"Could not parse boolean value: {value}")


def normalize_fuel_type(fuel_type: str) -> str:
	fuel = fuel_type.strip().lower()
	replacements = {
		"gasoline": "gas",
		"battery electric": "electric",
		"electric motor": "electric",
	}
	return replacements.get(fuel, fuel)


def build_feature_row(
	model_year: int,
	mileage_mi: float,
	fuel_type: str,
	brand: str,
	accident_reported: bool,
	clean_title_flag: bool,
	model: str = "Unknown",
	transmission: str = "Unknown",
	engine: str = "Unknown",
	ext_col: str = "Unknown",
	int_col: str = "Unknown",
) -> pd.DataFrame:
	normalized_fuel = normalize_fuel_type(fuel_type)

	car_age = CURRENT_YEAR - model_year
	if car_age < 0:
		car_age = np.nan

	feature_row = pd.DataFrame(
		[
			{
				"car_age": car_age,
				"scaled_mileage": mileage_mi / 1000,
				"fuel_type_normalized": normalized_fuel,
				"brand": brand,
				"model": model,
				"transmission": transmission,
				"engine": engine,
				"ext_col": ext_col,
				"int_col": int_col,
				"accident_reported": accident_reported,
				"clean_title_flag": clean_title_flag,
			}
		]
	)
	return feature_row


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Predict used-car price from user-provided vehicle details."
	)
	parser.add_argument(
		"--model-path",
		type=Path,
		default=Path("results/price_model.joblib"),
		help="Path to trained model artifact.",
	)
	parser.add_argument("--model-year", type=int, help="Vehicle model year.")
	parser.add_argument("--mileage", type=float, help="Vehicle mileage in miles.")
	parser.add_argument("--fuel-type", type=str, help="Fuel type (Gas, Diesel, Electric).")
	parser.add_argument("--brand", type=str, help="Brand name (for example: Toyota).")
	parser.add_argument("--vehicle-model", type=str, help="Vehicle model (for example: Camry).")
	parser.add_argument("--transmission", type=str, help="Transmission type (for example: Automatic).")
	parser.add_argument("--engine", type=str, help="Engine description (for example: 2.5L 4 Cylinder).")
	parser.add_argument("--ext-col", type=str, help="Exterior color (for example: White).")
	parser.add_argument("--int-col", type=str, help="Interior color (for example: Black).")
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


def prompt_if_missing(args: argparse.Namespace) -> argparse.Namespace:
	if args.model_year is None:
		args.model_year = int(input("Model year: ").strip())
	if args.mileage is None:
		args.mileage = float(input("Mileage (miles): ").strip())
	if not args.fuel_type:
		args.fuel_type = input("Fuel type (Gas/Diesel/Electric): ").strip()
	if not args.brand:
		args.brand = input("Brand: ").strip()
	if not args.vehicle_model:
		args.vehicle_model = input("Vehicle model (for example: Camry, leave blank to skip): ").strip() or "Unknown"
	if not args.transmission:
		args.transmission = input("Transmission (for example: Automatic, leave blank to skip): ").strip() or "Unknown"
	if not args.engine:
		args.engine = input("Engine (for example: 2.5L 4 Cylinder, leave blank to skip): ").strip() or "Unknown"
	if not args.ext_col:
		args.ext_col = input("Exterior color (leave blank to skip): ").strip() or "Unknown"
	if not args.int_col:
		args.int_col = input("Interior color (leave blank to skip): ").strip() or "Unknown"
	if args.accident_reported is None:
		args.accident_reported = input("Accident reported? (true/false): ").strip()
	if args.clean_title_flag is None:
		args.clean_title_flag = input("Clean title? (true/false): ").strip()
	return args


def main() -> None:
	args = parse_args()
	args = prompt_if_missing(args)

	if not args.model_path.exists():
		raise FileNotFoundError(
			f"Trained model not found: {args.model_path}. Run model_training.py first."
		)

	model = load(args.model_path)

	feature_row = build_feature_row(
		model_year=args.model_year,
		mileage_mi=args.mileage,
		fuel_type=args.fuel_type,
		brand=args.brand,
		accident_reported=parse_bool(args.accident_reported),
		clean_title_flag=parse_bool(args.clean_title_flag),
		model=args.vehicle_model or "Unknown",
		transmission=args.transmission or "Unknown",
		engine=args.engine or "Unknown",
		ext_col=args.ext_col or "Unknown",
		int_col=args.int_col or "Unknown",
	)

	predicted_price = model.predict(feature_row)[0]

	print("Input features:")
	print(feature_row.to_string(index=False))
	print(f"\nPredicted used-car price: ${predicted_price:,.2f}")


if __name__ == "__main__":
	main()
