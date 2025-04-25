notebooku:
	uv run jupyter notebook

tests:
	python -m unittest discover -v tests

testsu:
	uv run python -m unittest discover -vv tests