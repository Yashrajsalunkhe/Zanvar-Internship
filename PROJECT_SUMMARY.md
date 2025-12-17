# Project Summary - Zanvar Internship

## 🎯 Project Overview
Data analysis web application with AI-powered insights, built with React frontend and Go backend.

## 📂 Project Structure
```
Zanvar-Internship/
├── frontend/              # React + Vite frontend
│   ├── src/
│   │   ├── pages/        # Home, Upload, Chat, Profile
│   │   ├── api.js        # API utilities
│   │   └── App.jsx       # Main app component
│   ├── .env              # Development environment variables
│   ├── .env.production   # Production environment variables
│   └── package.json
│
├── backend/           # Go backend with Gin framework
│   ├── main.go          # Server and API routes
│   ├── analyzer.go      # Data analysis logic
│   ├── .env             # Development environment variables
│   ├── .env.production.example  # Production env template
│   ├── go.mod           # Go dependencies
│   ├── uploads/         # Uploaded files storage
│   └── generated_charts/ # Generated charts storage
│
├── build.sh             # Production build script
├── DEPLOYMENT_GUIDE.md  # Detailed deployment instructions
├── DEPLOYMENT_CHECKLIST.md  # Pre-deployment checklist
└── README.md            # Project documentation
```

## ✨ Features

### Frontend (React + Vite)
- **Home Page**: Landing page with hero section and feature highlights
- **Upload Page**: File upload interface supporting CSV, Excel, PDF, images
- **Chat Page**: Interactive AI chat for data analysis
- **Profile Page**: User settings and preferences
- **Responsive Design**: Tailwind CSS with modern UI/UX
- **Navigation**: Consistent header navigation across all pages

### Backend (Go + Gin)
- **File Upload**: Multi-format file handling with validation
- **AI Integration**: Google Gemini AI for intelligent data analysis
- **CSV Processing**: Automatic schema detection and data insights
- **Chat API**: Conversational interface for data queries
- **Chart Generation**: Dynamic visualization generation
- **CORS**: Configured for cross-origin requests

## 🛠️ Tech Stack

### Frontend
- **Framework**: React 18
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Routing**: React Router
- **HTTP Client**: Fetch API

### Backend
- **Language**: Go 1.20+
- **Web Framework**: Gin
- **AI SDK**: Google Generative AI Go SDK
- **File Processing**: CSV parsing, file validation
- **Middleware**: CORS, logging, recovery

## 🚀 Quick Start

### Development Mode

#### Start Backend
```bash
cd backend
cp .env.example .env
# Add your GEMINI_API_KEY to .env
go run main.go analyzer.go
```
Backend runs on: http://localhost:5000

#### Start Frontend
```bash
cd frontend
npm install
npm run dev
```
Frontend runs on: http://localhost:5173

### Production Build
```bash
./build.sh
```

This builds:
- Backend binary: `backend/server`
- Frontend static files: `frontend/dist/`

## 📋 API Endpoints

### Health Check
```
GET /
Response: {"status": "ok", "message": "...", "version": "1.0.0"}
```

### Upload File
```
POST /api/upload
Content-Type: multipart/form-data
Body: file (CSV, Excel, PDF, images)
Response: {"message": "...", "filename": "...", "file_info": {...}}
```

### Chat
```
POST /api/chat
Content-Type: application/json
Body: {"message": "Your question", "context": {...}}
Response: {"reply": "AI response", "response": "AI response"}
```

### Generate Chart
```
POST /api/generate-chart
Content-Type: application/json
Body: {"chart_type": "bar", "data": {...}}
Response: {"chart_url": "...", "status": "success"}
```

## 🔧 Configuration

### Environment Variables

#### Frontend (.env)
```
VITE_API_BASE=http://localhost:5000
```

#### Backend (.env)
```
GEMINI_API_KEY=your_gemini_api_key
PORT=5000
GIN_MODE=debug
```

### Production Configuration
- See `.env.production` files
- Update CORS origins in `backend/main.go`
- Set `GIN_MODE=release` for production

## 📦 Deployment

### Using Docker (Recommended)
```bash
docker-compose up -d
```

### Manual Deployment
1. Build both frontend and backend: `./build.sh`
2. Configure production environment variables
3. Deploy backend: `cd backend && GIN_MODE=release ./server`
4. Serve frontend static files with nginx/Apache
5. Configure SSL with Let's Encrypt

See `DEPLOYMENT_GUIDE.md` for detailed instructions.

## 🧪 Testing

### Test Backend Health
```bash
curl http://localhost:5000/
```

### Test File Upload
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@sample.csv"
```

### Test Chat
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Analyze my data"}'
```

## 🔒 Security Considerations

- ✅ File upload size limits (16MB)
- ✅ File type validation
- ✅ CORS restricted to allowed origins
- ✅ Environment variables for sensitive data
- ✅ Input validation on all endpoints
- ⚠️ Add rate limiting for production
- ⚠️ Implement authentication for production
- ⚠️ Regular API key rotation

## 📊 File Support

### Fully Supported (with AI analysis)
- **CSV**: Schema detection, data insights, column analysis

### Basic Support (upload only)
- **Excel**: .xlsx, .xls
- **PDF**: Document upload
- **Images**: .jpg, .jpeg, .png, .gif

## 🐛 Troubleshooting

### Backend Issues
- **Port already in use**: Kill existing process on port 5000
  ```bash
  lsof -ti:5000 | xargs kill -9
  ```
- **API quota exceeded**: Check Gemini API usage in Google Cloud Console
- **CORS errors**: Verify frontend URL in CORS configuration

### Frontend Issues
- **API connection failed**: Check backend is running on correct port
- **Build errors**: Clear node_modules and reinstall
  ```bash
  rm -rf node_modules package-lock.json
  npm install
  ```

## 📝 Development Notes

### Recent Improvements
- ✅ Fixed Go backend response structure (dual reply/response fields)
- ✅ Updated all page navigation to be consistent
- ✅ Added hover effects to all interactive elements
- ✅ Cleaned up project structure (removed unused files)
- ✅ Created comprehensive deployment documentation
- ✅ Added production build script

### Known Limitations
- API quota limits on Gemini API (depends on your plan)
- File processing limited to CSV for detailed analysis
- No user authentication (planned for future release)
- No database persistence (files stored on disk)

## 🔮 Future Enhancements
- [ ] User authentication and authorization
- [ ] Database integration for persistent storage
- [ ] Advanced chart customization
- [ ] Excel file detailed analysis
- [ ] PDF text extraction and analysis
- [ ] Image analysis with vision AI
- [ ] Real-time collaboration features
- [ ] Export analysis reports

## 📄 License
[Your License Here]

## 👥 Contributors
Yashraj - Developer

## 📞 Support
For issues or questions:
1. Check troubleshooting section
2. Review deployment guide
3. Check backend logs: `journalctl -u zanvar-backend`
4. Check frontend console in browser DevTools

## 🙏 Acknowledgments
- Google Gemini AI for intelligent analysis
- Gin framework for robust Go web server
- React and Vite for modern frontend development
- Tailwind CSS for beautiful styling
