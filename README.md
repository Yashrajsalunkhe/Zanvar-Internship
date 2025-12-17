# 📊 Zanvar Data Insights

<div align="center">

![Status](https://img.shields.io/badge/status-production_ready-success?style=for-the-badge&logo=statuspage)
![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge&logo=open-source-initiative)
![Go](https://img.shields.io/badge/Go-1.20+-00ADD8?style=for-the-badge&logo=go)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react)
![Vite](https://img.shields.io/badge/Vite-PRO-646CFF?style=for-the-badge&logo=vite)
![Tailwind](https://img.shields.io/badge/Tailwind_CSS-Premium-38B2AC?style=for-the-badge&logo=tailwind-css)

<br/>

**Unlock the hidden value in your data with AI-powered analytics.**

[Features](#-key-features) • [Tech Stack](#-tech-stack) • [Getting Started](#-getting-started) • [Usage](#-usage-guide) • [API](#-api-reference)

</div>

---

## 📖 Overview

**Zanvar Data Insights** is a state-of-the-art data analysis platform that transforms raw spreadsheets into actionable intelligence. By combining a high-performance **Go backend** with a stunning, **premium React frontend**, we offer users a seamless experience to analyzing complex datasets.

Interact with your data using natural language through our **AI Chatbot**, visualize trends with **dynamic charting**, and enjoy a modern, **glassmorphism-inspired UI** that feels as good as it looks.

## ✨ Key Features

### 🧠 Intelligent Analysis
- **AI Chatbot**: Ask questions in plain English and get instant, data-backed answers using Google Gemini AI.
- **Smart Insights**: Automatically detect schemas and generate summary statistics upon upload.

### 🎨 Premium User Experience
- **Modern UI/UX**: Deep dark mode with ambient lighting and glassmorphism effects.
- **Responsive Design**: Fully optimized for desktop and tablet experiences.
- **Interactive Visualizations**: Beautiful, animated bar, line, and pie charts generated on the fly.

### 🚀 High Performance
- **Fast Processing**: Native Go implementation for CSV/Excel parsing handles large files with ease.
- **Concurrent Architecture**: Built to handle multiple requests simultaneously without lag.

## 🛠️ Tech Stack

### Frontend
- **Framework**: React 18
- **Build Tool**: Vite (Lightning fast HMR)
- **Styling**: Tailwind CSS (Custom Design System)
- **Routing**: React Router v6
- **Typography**: Inter & Outfit (Google Fonts)

### Backend
- **Language**: Go (Golang) 1.20+
- **Framework**: Gin Gonic
- **AI Engine**: Google Gemini Pro
- **Data Processing**: Native encoding/csv and excelize

## 🚀 Getting Started

Follow these steps to set up the project locally.

### Prerequisites
- **Node.js** (v18 or higher)
- **Go** (v1.20 or higher)
- **Git**
- A **Google Gemini API Key** ([Get it here](https://makersuite.google.com/app/apikey))

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Aditya-padale/Zanvar-Internship.git
    cd Zanvar-Internship
    ```

2.  **Backend Setup**
    ```bash
    cd backend-go
    # Linux/Mac
    cp .env.example .env 
    # Windows (PowerShell)
    # copy .env.example .env
    
    # Open .env and add your GEMINI_API_KEY
    go mod download
    go run main.go analyzer.go
    ```
    > Server starts at `http://localhost:5000`

3.  **Frontend Setup**
    ```bash
    cd ../frontend
    npm install
    npm run dev
    ```
    > App runs at `http://localhost:5173`

## 🎯 Usage Guide

### 1. Upload Your Data
Navigate to the **Upload** page. Drag and drop your `.csv` or `.xlsx` file into the glass drop zone. Watch the progress bar as our system instantly processes your dataset.

### 2. Chat with AI
Once uploaded, you'll be redirected to the **Chat** interface.
- **Ask**: "What is the total revenue for last quarter?"
- **Analyze**: "Show me the top 5 performing regions."
- **Visualize**: "Create a bar chart for sales by category."

### 3. Explore Insights
Visit your **Profile** to manage settings or return to the **Home** dashboard for a quick overview of capabilities.

## 📋 API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | API Health Check |
| `/api/upload` | `POST` | Upload and process a file (Multipart form) |
| `/api/chat` | `POST` | Send a prompt to the AI agent |
| `/api/generate-chart` | `POST` | Request a chart configuration |

## 📂 Project Structure

```bash
Zanvar-Internship/
├── frontend/             # React Application
│   ├── src/
│   │   ├── pages/        # Premium UI Pages (Home, Chat, Upload)
│   │   ├── components/   # Reusable Glass Components
│   │   └── api.js        # Backend Integration
│   └── tailwind.config.js
│
├── backend-go/           # Go API Server
│   ├── main.go           # Entry point & Routes
│   ├── analyzer.go       # Core Data Logic
│   └── uploads/          # Temporary File Storage
│
└── README.md             # Project Documentation
```

## 🔐 Security & Performance

- **Data Privacy**: Files are processed locally/in-memory where possible and strictly validated.
- **Rate Limiting**: API endpoints are protected against abuse.
- **Optimized Builds**: Frontend assets are minified and compressed (Gzip) for production.

## 🤝 Contributing

We welcome contributions!
1.  Fork the project.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

**Built with ❤️ for the Zanvar Internship**
<br/>
Developed by Yashraj

</div>
