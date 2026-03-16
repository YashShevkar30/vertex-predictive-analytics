.PHONY: install ingest cluster train score evaluate test

install:
	pip install -r requirements.txt

generate:
	python -m vertex.data.generator

ingest:
	python -m vertex.spark.ingestion

cluster:
	python -m vertex.features.clustering

train:
	python -m vertex.models.train

score:
	python -m vertex.scoring.batch_scorer

evaluate:
	python -m vertex.evaluation.evaluate

test:
	pytest tests/ -v --tb=short

pipeline:
	make generate ingest cluster train evaluate score
