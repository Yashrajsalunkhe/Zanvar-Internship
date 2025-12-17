#!/bin/bash

# Test the Go backend API endpoints

echo "🧪 Testing Go Backend API..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Base URL
BASE_URL="http://localhost:5000"

# Test 1: Health Check
echo "1️⃣  Testing Health Check..."
response=$(curl -s "$BASE_URL/")
if echo "$response" | grep -q "status"; then
    echo -e "${GREEN}✅ Health check passed${NC}"
    echo "$response" | jq '.' 2>/dev/null || echo "$response"
else
    echo -e "${RED}❌ Health check failed${NC}"
fi
echo ""

# Test 2: Chat Endpoint
echo "2️⃣  Testing Chat Endpoint..."
response=$(curl -s -X POST "$BASE_URL/api/chat" \
    -H "Content-Type: application/json" \
    -d '{"message": "Hello, can you help me analyze data?"}')
if echo "$response" | grep -q "reply"; then
    echo -e "${GREEN}✅ Chat endpoint passed${NC}"
    echo "$response" | jq '.reply' 2>/dev/null || echo "$response"
else
    echo -e "${RED}❌ Chat endpoint failed${NC}"
    echo "$response"
fi
echo ""

# Test 3: File Upload (requires a test file)
if [ -f "../sample_data/sample.csv" ]; then
    echo "3️⃣  Testing File Upload..."
    response=$(curl -s -X POST "$BASE_URL/api/upload" \
        -F "file=@../sample_data/sample.csv")
    if echo "$response" | grep -q "File uploaded successfully"; then
        echo -e "${GREEN}✅ File upload passed${NC}"
        echo "$response" | jq '.' 2>/dev/null || echo "$response"
    else
        echo -e "${RED}❌ File upload failed${NC}"
        echo "$response"
    fi
else
    echo "3️⃣  Skipping file upload test (no test file found)"
fi
echo ""

echo "🎉 Testing complete!"
