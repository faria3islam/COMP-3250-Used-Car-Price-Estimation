from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def clean_used_car_data(df: pd.DataFrame) -> pd.DataFrame:
	"""Apply the project cleaning logic to the raw used-car dataset."""
	clean = df.copy()

	clean.columns = [c.strip().lower().replace(" ", "_") for c in clean.columns]

	for col in clean.select_dtypes(include="object").columns:
		clean[col] = clean[col].astype(str).str.strip()

	clean = clean.replace({"nan": np.nan, "None": np.nan})

	placeholders = ["–", "not supported", "N/A", ""]
	clean = clean.replace(placeholders, np.nan)

	brand_map = {
		"Land": "Land Rover",
		"Aston": "Aston Martin",
		"Alfa": "Alfa Romeo",
		"smart": "Smart",
	}
	if "brand" in clean.columns:
		clean["brand"] = clean["brand"].replace(brand_map)

	for col in ["ext_col", "int_col"]:
		if col in clean.columns:
			clean[col] = clean[col].str.replace(r"\.+$", "", regex=True).str.strip()

	if "price" in clean.columns:
		clean["price_usd"] = pd.to_numeric(
			clean["price"].replace(r"[\$,]", "", regex=True),
			errors="coerce",
		)

	if "milage" in clean.columns:
		clean["mileage_mi"] = pd.to_numeric(
			clean["milage"].replace(r"[^0-9]", "", regex=True),
			errors="coerce",
		)

	if "clean_title" in clean.columns:
		clean["clean_title_flag"] = clean["clean_title"].map({"Yes": True})

	if "accident" in clean.columns:
		clean["accident_reported"] = clean["accident"].map(
			{
				"At least 1 accident or damage reported": True,
				"None reported": False,
			}
		)

	clean = clean.drop_duplicates().reset_index(drop=True)
	return clean


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Clean the used-cars dataset and export a model-ready CSV."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("data/used_cars.csv"),
		help="Path to raw input CSV.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("data/cleaned_used_cars.csv"),
		help="Path for cleaned output CSV.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if not args.input.exists():
		raise FileNotFoundError(f"Input file not found: {args.input}")

	raw = pd.read_csv(args.input)
	clean = clean_used_car_data(raw)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	clean.to_csv(args.output, index=False)

	print(f"Raw shape: {raw.shape}")
	print(f"Cleaned shape: {clean.shape}")
	print(f"Saved cleaned dataset to: {args.output.resolve()}")


if __name__ == "__main__":
	main()
