.PHONY: install test train research predict

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

train:
	python -m pitch_type_prediction.train --data data/pitch-type-prediction-data.csv --output-dir artifacts/reproduced

research:
	python -m pitch_type_prediction.research --data data/pitch-type-prediction-data.csv --output artifacts/research_rerun.csv

predict:
	python -m pitch_type_prediction.predict --model artifacts/pitch_type_model.joblib --input demo_input.csv --output demo_predictions.csv
