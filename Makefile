lint:
	pylint src tests

test:
	pytest tests

.PHONY: lint test
