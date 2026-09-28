.PHONY: install test train train-notebook

install:
	pip install -r requirements.txt

test:
	pytest -q

train:            ## regularised XGBoost -> submissions/submission.csv
	PYTHONPATH=src python -m purchase_value.train --preset regularised

train-notebook:   ## settings behind the notebook's submission (deep trees + RFE 25)
	PYTHONPATH=src python -m purchase_value.train --preset notebook --rfe 25
