#!/bin/bash

# Zanvar Backend Go - Setup Script

echo "🔧 Setting up Zanvar Go Backend..."

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo "❌ Go is not installed. Please install Go first."
    echo "Visit: https://golang.org/doc/install"
    exit 1
fi

echo "✅ Go version: $(go version)"

# Navigate to backend-go directory
cd "$(dirname "$0")/backend-go" || exit 1

# Copy .env.example to .env if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created .env file. Please update it with your API keys."
else
    echo "✅ .env file already exists"
fi

# Install dependencies
echo "📦 Installing Go dependencies..."
go get -u github.com/gin-gonic/gin
go get -u github.com/gin-contrib/cors
go get -u github.com/joho/godotenv
go get -u github.com/google/generative-ai-go/genai
go get -u google.golang.org/api/option

# Tidy up dependencies
go mod tidy

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "  1. Update backend-go/.env with your GOOGLE_API_KEY"
echo "  2. Run: ./run-go.sh"
echo ""
