# Makefile for Go Backend

.PHONY: help setup run build test clean deps

# Default target
help:
	@echo "Zanvar Go Backend - Available Commands:"
	@echo ""
	@echo "  make setup    - Install dependencies and setup environment"
	@echo "  make run      - Run the Go server"
	@echo "  make build    - Build production binary"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Clean build artifacts"
	@echo "  make deps     - Download and tidy dependencies"
	@echo ""

# Setup environment and install dependencies
setup:
	@echo "🔧 Setting up Go backend..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "📝 Created .env file"; fi
	@mkdir -p uploads generated_charts
	@go mod download
	@go mod tidy
	@echo "✅ Setup complete!"

# Download dependencies
deps:
	@echo "📦 Downloading dependencies..."
	@go mod download
	@go mod tidy

# Run the server
run:
	@echo "🚀 Starting server..."
	@mkdir -p uploads generated_charts
	@go run main.go analyzer.go

# Build production binary
build:
	@echo "🔨 Building production binary..."
	@go build -o zanvar-server main.go analyzer.go
	@echo "✅ Binary created: ./zanvar-server"

# Run tests
test:
	@echo "🧪 Running tests..."
	@go test ./... -v

# Clean build artifacts
clean:
	@echo "🧹 Cleaning..."
	@rm -f zanvar-server
	@rm -rf uploads/* generated_charts/*
	@echo "✅ Cleaned!"

# Development mode with auto-reload (requires air)
dev:
	@if command -v air > /dev/null; then \
		air; \
	else \
		echo "⚠️  'air' not installed. Install with: go install github.com/air-verse/air@latest"; \
		echo "Running without auto-reload..."; \
		make run; \
	fi
