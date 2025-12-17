# Zanvar Backend - Go Edition

A high-performance backend service built with Go (Golang) for intelligent data analysis and AI-powered chat.

## Features

- 🚀 **High Performance**: Built with Go for maximum speed and concurrency
- 📊 **Data Analysis**: CSV file processing and statistical analysis
- 🤖 **AI Integration**: Google Gemini AI for intelligent chat responses
- 📁 **File Upload**: Support for CSV, PDF, images, and text files
- 📈 **Chart Generation**: Data visualization capabilities
- 🔒 **CORS Enabled**: Secure cross-origin requests

## Tech Stack

- **Go 1.21+**
- **Gin Web Framework**: Fast HTTP router
- **Google Generative AI**: For AI-powered responses
- **CSV Processing**: Built-in data analysis

## Prerequisites

- Go 1.21 or higher
- Google API Key (for Gemini AI)

## Setup

1. **Run the setup script**:
   ```bash
   chmod +x setup-go.sh
   ./setup-go.sh
   ```

2. **Configure environment variables**:
   Edit `backend-go/.env` and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_actual_api_key_here
   PORT=5000
   ```

3. **Start the server**:
   ```bash
   chmod +x run-go.sh
   ./run-go.sh
   ```

   Or manually:
   ```bash
   cd backend-go
   go run main.go analyzer.go
   ```

## API Endpoints

### Health Check
```
GET /
```

### Upload File
```
POST /api/upload
Content-Type: multipart/form-data

Body: file (binary)
```

### Chat
```
POST /api/chat
Content-Type: application/json

{
  "message": "Your question here"
}
```

### Generate Chart
```
POST /api/generate-chart
Content-Type: application/json

{
  "type": "bar",
  "data": {...}
}
```

## Project Structure

```
backend-go/
├── main.go          # Main server and API handlers
├── analyzer.go      # Data analysis utilities
├── go.mod           # Go module dependencies
├── go.sum           # Dependency checksums
├── .env             # Environment variables
├── uploads/         # Uploaded files directory
└── generated_charts/# Generated chart files
```

## Development

### Install Dependencies
```bash
cd backend-go
go mod download
```

### Run Tests
```bash
go test ./...
```

### Build for Production
```bash
go build -o server main.go analyzer.go
./server
```

## Supported File Types

- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- PDF (`.pdf`)
- Images (`.png`, `.jpg`, `.jpeg`, `.gif`)
- Text (`.txt`)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | Required |
| `PORT` | Server port | 5000 |

## Performance Benefits

Compared to Python Flask:
- **10-100x faster** request handling
- **Lower memory footprint**
- **Better concurrency** with goroutines
- **Faster CSV processing**
- **Native compilation** for deployment

## Migrating from Python Backend

The Go backend maintains API compatibility with the Python version. Simply:

1. Update frontend `VITE_API_BASE` if needed
2. Stop Python backend
3. Start Go backend on the same port

## Troubleshooting

### Port already in use
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9
```

### Missing dependencies
```bash
go mod tidy
go mod download
```

### AI not responding
- Verify `GOOGLE_API_KEY` is set correctly in `.env`
- Check API key has Gemini API access enabled

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details
