# Convenience targets — run from the repo root with the venv activated.

.PHONY: install data eda train test lint api mlflow-ui docker-build docker-run compose-up k8s-deploy

install:
	pip install -r requirements.txt

data:
	python -m src.data.download
	python -m src.data.preprocess

eda:
	python -m src.eda

train:
	python -m src.models.train

test:
	pytest -v

lint:
	ruff check src tests

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

docker-build:
	docker build -t heart-disease-api:latest .

docker-run:
	docker run --rm -p 8000:8000 heart-disease-api:latest

compose-up:
	docker compose up --build

k8s-deploy:
	kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
