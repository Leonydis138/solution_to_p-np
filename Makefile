.PHONY: help install test lint format clean run docker-build docker-run

help:
	@echo "Circuit Lower Bounds Research Platform"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linting"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make run          - Run Streamlit app"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/ streamlit_app.py
	isort src/ tests/ streamlit_app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info htmlcov/ .pytest_cache/ .mypy_cache/

run:
	streamlit run streamlit_app.py

docker-build:
	docker build -t circuit-lower-bounds .

docker-run:
	docker-compose up
