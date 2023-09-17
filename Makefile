lint:
	pylint src tests app.py

test:
	pytest tests

format:
	@echo "Running black..."
	@black .
	@echo "Running isort..."
	@isort .
	@echo "Formatting complete!"

.PHONY: lint test format
