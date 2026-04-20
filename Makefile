.PHONY: install test train serve format

install:
	pip install -r requirements.txt

test:
	pytest tests/

train:
	python src/train.py

serve:
	uvicorn main:app --reload

format:
	black src/ tests/ main.py
