lint:
	pylint src tests

test:
	pytest tests

format:
	@echo "Running isort..."
	@isort .
	@echo "Running black..."
	@black .
	@echo "Formatting complete!"

.PHONY: lint test format
