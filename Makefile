.PHONY: help build run clean up down logs test

# Default target
help:
	@echo "MLOps Pipeline - Available Commands:"
	@echo "  make build      - Build all Docker images"
	@echo "  make build-data - Build data fetch component"
	@echo "  make build-preprocess - Build preprocessing component"
	@echo "  make build-train - Build training component"
	@echo "  make build-eval - Build evaluation component"
	@echo "  make build-registry - Build model registry component"
	@echo "  make run        - Run all components with docker-compose"
	@echo "  make up         - Start services in background"
	@echo "  make down       - Stop all services"
	@echo "  make logs       - View logs"
	@echo "  make clean      - Clean Docker images and containers"
	@echo "  make test       - Run basic tests"

# Build all components
build:
	docker-compose build

# Build individual components
build-data:
	docker build -f components/data_fetcher/Dockerfile -t mlops-data-fetch .

build-preprocess:
	docker build -f components/preprocess/Dockerfile -t mlops-preprocess .

build-train:
	docker build -f components/train/Dockerfile -t mlops-train .

build-eval:
	docker build -f components/evaluate/Dockerfile -t mlops-evaluate .

build-registry:
	docker build -f components/model_registry/Dockerfile -t mlops-registry .

# Run with docker-compose
run:
	docker-compose up

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

# Clean up Docker images and containers
clean:
	docker-compose down -v
	docker rmi mlops-data-fetch mlops-preprocess mlops-train mlops-evaluate mlops-registry 2>/dev/null || true
	docker system prune -f

# Run basic tests
test:
	@echo "Running basic component tests..."
	python -c "import sys; sys.path.insert(0, 'components/data_fetcher'); from data_fetch import DataFetcher; print('✓ data_fetch.py')"
	python -c "import sys; sys.path.insert(0, 'components/preprocess'); from preprocess import DataPreprocessor; print('✓ preprocess.py')"
	python -c "import sys; sys.path.insert(0, 'components/evaluate'); from evaluate import ModelEvaluator; print('✓ evaluate.py')"
	python -c "import sys; sys.path.insert(0, 'components/model_registry'); from model_registry import ModelRegistry; print('✓ model_registry.py')"
	@echo "All tests passed!"

