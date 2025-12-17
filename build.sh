#!/bin/bash

# Build script for Zanvar Internship Project
# This script builds both frontend and backend for production deployment

set -e

echo "🚀 Starting build process..."

# Build Backend
echo ""
echo "🔨 Building Go backend..."
cd backend
echo "   - Downloading dependencies..."
go mod download
echo "   - Compiling binary..."
go build -o server main.go analyzer.go
echo "   ✅ Backend build complete: backend/server"

# Build Frontend
echo ""
echo "📦 Building React Frontend..."
cd ../frontend
echo "   - Installing dependencies..."
npm install
echo "   - Creating production build..."
npm run build
echo "   ✅ Frontend build complete: frontend/dist/"

echo ""
echo "🎉 Build complete! Ready for deployment."
echo ""
echo "Next steps:"
echo "1. Configure production environment variables:"
echo "   - backend/.env.production (copy from .env.production.example)"
echo "   - frontend/.env.production (update API URL)"
echo ""
echo "2. Deploy backend:"
echo "   cd backend && GIN_MODE=release ./server"
echo ""
echo "3. Deploy frontend:"
echo "   - Serve frontend/dist/ with nginx/Apache"
echo "   - Or use: cd frontend && npm run preview"
echo ""
echo "📖 See DEPLOYMENT_GUIDE.md for detailed instructions"
