#!/bin/bash

# Zanvar Backend Go - Run Script

echo "🚀 Starting Zanvar Go Backend..."

# Navigate to backend-go directory
cd "$(dirname "$0")/backend-go" || exit 1

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "📝 Please update .env with your API keys"
fi

# Create necessary directories
mkdir -p uploads
mkdir -p generated_charts

# Run the Go application
echo "🔥 Running Go server..."
go run main.go analyzer.go
