package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/google/generative-ai-go/genai"
	"github.com/joho/godotenv"
	"google.golang.org/api/option"
)

const (
	MaxFileSize  = 16 * 1024 * 1024 // 16MB
	UploadFolder = "uploads"
	ChartsFolder = "generated_charts"
)

var (
	allowedExtensions = map[string]bool{
		".txt":  true,
		".pdf":  true,
		".png":  true,
		".jpg":  true,
		".jpeg": true,
		".gif":  true,
		".csv":  true,
		".xlsx": true,
		".xls":  true,
	}
	genAIClient *genai.Client
	genAIModel  *genai.GenerativeModel

	// Store last uploaded file info and data for chat context
	lastUploadedFile *FileInfo
	lastCSVData      *CSVInfo
)

// ConversationMemory stores context across chat messages
type ConversationMemory struct {
	LastQuestion     string                 `json:"last_question"`
	LastAnswer       string                 `json:"last_answer"`
	MentionedParts   []string               `json:"mentioned_parts"`
	MentionedDates   []string               `json:"mentioned_dates"`
	MentionedDefects []string               `json:"mentioned_defects"`
	SessionContext   map[string]interface{} `json:"session_context"`
	Context          map[string]interface{} `json:"context"`
}

var conversationMemory = &ConversationMemory{
	MentionedParts:   make([]string, 0),
	MentionedDates:   make([]string, 0),
	MentionedDefects: make([]string, 0),
	SessionContext:   make(map[string]interface{}),
	Context:          make(map[string]interface{}),
}

// FileInfo represents uploaded file information
type FileInfo struct {
	Filename    string                 `json:"filename"`
	Size        int64                  `json:"size"`
	Type        string                 `json:"type"`
	UploadedAt  time.Time              `json:"uploaded_at"`
	ProcessedAt time.Time              `json:"processed_at,omitempty"`
	Info        map[string]interface{} `json:"info,omitempty"`
}

// ChatRequest represents a chat message request
type ChatRequest struct {
	Message string `json:"message" binding:"required"`
}

// ChatResponse represents a chat message response
type ChatResponse struct {
	Reply          string                 `json:"reply"`
	Response       string                 `json:"response"` // For frontend compatibility
	ChartData      interface{}            `json:"chart_data,omitempty"`
	Insights       []string               `json:"insights,omitempty"`
	Suggestions    []string               `json:"suggestions,omitempty"`
	AdditionalData map[string]interface{} `json:"additional_data,omitempty"`
}

// CSVInfo represents CSV file analysis
type CSVInfo struct {
	Shape          []int                    `json:"shape"`
	Columns        []string                 `json:"columns"`
	DataTypes      map[string]string        `json:"data_types"`
	NullCounts     map[string]int           `json:"null_counts"`
	NumericSummary map[string]interface{}   `json:"numeric_summary,omitempty"`
	Preview        []map[string]interface{} `json:"preview"`
}

func init() {
	// Load environment variables
	if err := godotenv.Load(); err != nil {
		log.Println("Warning: .env file not found")
	}

	// Create necessary directories
	os.MkdirAll(UploadFolder, 0755)
	os.MkdirAll(ChartsFolder, 0755)

	// Initialize Google Generative AI
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey != "" {
		ctx := context.Background()
		var err error
		genAIClient, err = genai.NewClient(ctx, option.WithAPIKey(apiKey))
		if err != nil {
			log.Printf("Warning: Failed to initialize Google AI client: %v\n", err)
		} else {
			genAIModel = genAIClient.GenerativeModel("gemini-2.5-flash")
			log.Println("✅ Google Gemini 2.5 Flash configured successfully")
		}
	} else {
		log.Println("Warning: GOOGLE_API_KEY not found in environment variables")
	}

	// Auto-load the most recent CSV file for chat context
	loadMostRecentCSV()
}

func main() {
	router := gin.Default()

	// CORS configuration - reads from environment variable
	config := cors.DefaultConfig()
	
	// Get allowed origins from environment variable, fallback to localhost for development
	allowedOrigins := os.Getenv("ALLOWED_ORIGINS")
	if allowedOrigins != "" {
		// Split multiple origins by comma if provided
		config.AllowOrigins = strings.Split(allowedOrigins, ",")
		log.Printf("CORS: Allowing origins: %v", config.AllowOrigins)
	} else {
		// Default to localhost for development
		config.AllowOrigins = []string{"http://localhost:5173", "http://localhost:3000"}
		log.Println("CORS: Using default localhost origins")
	}
	
	config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{"Origin", "Content-Type", "Accept", "Authorization"}
	config.AllowCredentials = true
	router.Use(cors.New(config))

	// Routes
	router.GET("/", healthCheck)
	router.POST("/api/upload", uploadFile)
	router.POST("/api/chat", handleChat)
	router.POST("/api/generate-chart", generateChart)

	// Start server
	port := os.Getenv("PORT")
	if port == "" {
		port = "5000"
	}

	log.Printf("🚀 Server starting on port %s\n", port)
	if err := router.Run(":" + port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

func healthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "ok",
		"message":   "Zanvar Backend API - Go Edition",
		"version":   "1.0.0",
		"timestamp": time.Now().Format(time.RFC3339),
	})
}

func uploadFile(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No file uploaded"})
		return
	}

	// Check file size
	if file.Size > MaxFileSize {
		c.JSON(http.StatusBadRequest, gin.H{"error": "File too large (max 16MB)"})
		return
	}

	// Validate file extension
	ext := strings.ToLower(filepath.Ext(file.Filename))
	if !allowedExtensions[ext] {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": fmt.Sprintf("File type not allowed. Allowed types: %v", getAllowedExtensions()),
		})
		return
	}

	// Save file
	filename := fmt.Sprintf("%d_%s", time.Now().Unix(), file.Filename)
	filepath := filepath.Join(UploadFolder, filename)

	if err := c.SaveUploadedFile(file, filepath); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save file"})
		return
	}

	// Process file based on type
	fileInfo := FileInfo{
		Filename:   filename,
		Size:       file.Size,
		Type:       ext,
		UploadedAt: time.Now(),
	}

	// Process CSV files
	if ext == ".csv" {
		info, err := processCSVFile(filepath)
		if err != nil {
			log.Printf("Error processing CSV: %v\n", err)
		} else {
			fileInfo.Info = map[string]interface{}{
				"csv_info": info,
			}
			fileInfo.ProcessedAt = time.Now()

			// Store globally for chat context
			lastUploadedFile = &fileInfo
			lastCSVData = info
			log.Printf("Stored CSV data for chat context: %d rows, %d columns\n", info.Shape[0], info.Shape[1])
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"message":   "File uploaded successfully",
		"filename":  filename,
		"file_info": fileInfo,
		"status":    "success",
	})
}

func handleChat(c *gin.Context) {
	var req ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	// Update conversation memory
	conversationMemory.LastQuestion = req.Message

	// Generate response using Google AI
	response, err := generateAIResponse(req.Message)
	if err != nil {
		// Log the error for debugging
		log.Printf("AI Response Error: %v\n", err)
		// Fallback response
		fallbackMsg := fmt.Sprintf("I received your message: '%s'. AI service is currently unavailable. Please try again later.", req.Message)
		response = &ChatResponse{
			Reply:    fallbackMsg,
			Response: fallbackMsg, // For frontend compatibility
			Suggestions: []string{
				"Upload a CSV file to analyze data",
				"Ask about data insights",
				"Request chart generation",
			},
		}
	}

	conversationMemory.LastAnswer = response.Reply

	c.JSON(http.StatusOK, response)
}

func generateChart(c *gin.Context) {
	var req map[string]interface{}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	chartType, _ := req["type"].(string)
	data, _ := req["data"].(map[string]interface{})

	// For now, return a placeholder response
	// You can integrate with plotly or other charting libraries
	c.JSON(http.StatusOK, gin.H{
		"status":     "success",
		"chart_type": chartType,
		"data":       data,
		"message":    "Chart generation endpoint - implementation pending",
	})
}

func generateAIResponse(message string) (*ChatResponse, error) {
	if genAIModel == nil {
		return nil, fmt.Errorf("AI model not initialized")
	}

	ctx := context.Background()

	// Build prompt with context
	prompt := buildPromptWithContext(message)

	resp, err := genAIModel.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return nil, err
	}

	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return nil, fmt.Errorf("no response from AI")
	}

	// Extract text from response
	var replyText string
	for _, part := range resp.Candidates[0].Content.Parts {
		replyText += fmt.Sprintf("%v", part)
	}

	response := &ChatResponse{
		Reply:    replyText,
		Response: replyText, // For frontend compatibility
		Suggestions: []string{
			"Tell me more about the data",
			"Generate a visualization",
			"Show me statistical insights",
		},
	}

	// Check if user is requesting a chart and we have data
	chartData := detectAndGenerateChart(message)
	if chartData != nil {
		response.ChartData = chartData
	}

	return response, nil
}

func detectAndGenerateChart(message string) map[string]interface{} {
	// Check if message contains chart-related keywords
	messageLower := strings.ToLower(message)
	chartKeywords := []string{"chart", "graph", "plot", "visualize", "visualization", "bar chart", "pie chart", "line chart", "histogram"}

	isChartRequest := false
	for _, keyword := range chartKeywords {
		if strings.Contains(messageLower, keyword) {
			isChartRequest = true
			break
		}
	}

	if !isChartRequest || lastCSVData == nil {
		return nil
	}

	// Determine chart type based on keywords
	chartType := "bar"
	if strings.Contains(messageLower, "pie") {
		chartType = "pie"
	} else if strings.Contains(messageLower, "line") {
		chartType = "line"
	} else if strings.Contains(messageLower, "scatter") {
		chartType = "scatter"
	}

	// Generate chart data based on CSV columns
	// Find the first categorical and numeric columns
	var categoryCol string
	var valueCol string

	for col, dtype := range lastCSVData.DataTypes {
		if dtype == "string" && categoryCol == "" {
			categoryCol = col
		}
		if dtype == "numeric" && valueCol == "" {
			valueCol = col
		}
	}

	// If we don't have both, use first two columns
	if categoryCol == "" && len(lastCSVData.Columns) > 0 {
		categoryCol = lastCSVData.Columns[0]
	}
	if valueCol == "" && len(lastCSVData.Columns) > 1 {
		valueCol = lastCSVData.Columns[1]
	} else if valueCol == "" {
		valueCol = categoryCol // Use same column
	}

	// Extract data for chart
	labels := make([]string, 0)
	values := make([]interface{}, 0)

	for _, row := range lastCSVData.Preview {
		if label, ok := row[categoryCol]; ok {
			labels = append(labels, fmt.Sprintf("%v", label))
		}
		if val, ok := row[valueCol]; ok {
			values = append(values, val)
		}
	}

	// Return chart data in a format the frontend can use
	return map[string]interface{}{
		"type":   chartType,
		"title":  fmt.Sprintf("%s by %s", valueCol, categoryCol),
		"labels": labels,
		"datasets": []map[string]interface{}{
			{
				"label": valueCol,
				"data":  values,
			},
		},
		"xAxisLabel": categoryCol,
		"yAxisLabel": valueCol,
	}
}

func buildPromptWithContext(message string) string {
	context := ""
	if conversationMemory.LastQuestion != "" {
		context = fmt.Sprintf("Previous question: %s\nPrevious answer: %s\n\n",
			conversationMemory.LastQuestion, conversationMemory.LastAnswer)
	}

	// Add uploaded file context
	fileContext := ""
	if lastUploadedFile != nil && lastCSVData != nil {
		fileContext = fmt.Sprintf(`\n\nUPLOADED DATA CONTEXT:
- Filename: %s
- Format: CSV
- Dimensions: %d rows × %d columns
- Columns: %v
- Data Types: %v

Sample Data (first few rows):
`,
			lastUploadedFile.Filename,
			lastCSVData.Shape[0],
			lastCSVData.Shape[1],
			lastCSVData.Columns,
			lastCSVData.DataTypes)

		// Add preview data
		for i, row := range lastCSVData.Preview {
			fileContext += fmt.Sprintf("Row %d: %v\n", i+1, row)
		}

		fileContext += "\nUse this data to answer questions about the uploaded file. Provide specific insights based on the actual data shown above.\n"
	}

	prompt := fmt.Sprintf(`%s%sYou are an intelligent data analyst assistant. 
Respond to the following query in a helpful and informative way:

User Query: %s

Provide a clear, concise response. If the query is about data analysis, offer specific insights based on the uploaded data.
If it's about visualization, suggest appropriate chart types based on the columns and data types available.
When answering questions about the data, reference specific columns, values, and patterns you see in the sample data.`, context, fileContext, message)

	return prompt
}

func processCSVFile(filepath string) (*CSVInfo, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	if len(records) == 0 {
		return nil, fmt.Errorf("empty CSV file")
	}

	headers := records[0]
	dataRows := records[1:]

	// Build CSV info
	info := &CSVInfo{
		Shape:      []int{len(dataRows), len(headers)},
		Columns:    headers,
		DataTypes:  make(map[string]string),
		NullCounts: make(map[string]int),
		Preview:    make([]map[string]interface{}, 0),
	}

	// Infer data types and count nulls
	for i, header := range headers {
		info.DataTypes[header] = inferDataType(dataRows, i)
		info.NullCounts[header] = countNulls(dataRows, i)
	}

	// Generate preview (first 5 rows)
	previewLimit := 5
	if len(dataRows) < previewLimit {
		previewLimit = len(dataRows)
	}

	for i := 0; i < previewLimit; i++ {
		row := make(map[string]interface{})
		for j, header := range headers {
			if j < len(dataRows[i]) {
				row[header] = dataRows[i][j]
			} else {
				row[header] = nil
			}
		}
		info.Preview = append(info.Preview, row)
	}

	return info, nil
}

func inferDataType(rows [][]string, colIndex int) string {
	// Simple type inference - can be enhanced
	if len(rows) == 0 {
		return "unknown"
	}

	// Check first non-null value
	for _, row := range rows {
		if colIndex >= len(row) || row[colIndex] == "" {
			continue
		}

		value := row[colIndex]

		// Try to parse as number
		var f float64
		if _, err := fmt.Sscanf(value, "%f", &f); err == nil {
			return "numeric"
		}

		// Default to string
		return "string"
	}

	return "unknown"
}

func countNulls(rows [][]string, colIndex int) int {
	count := 0
	for _, row := range rows {
		if colIndex >= len(row) || row[colIndex] == "" {
			count++
		}
	}
	return count
}

func getAllowedExtensions() []string {
	exts := make([]string, 0, len(allowedExtensions))
	for ext := range allowedExtensions {
		exts = append(exts, ext)
	}
	return exts
}

// loadMostRecentCSV loads the most recent CSV file from uploads folder on server start
func loadMostRecentCSV() {
	files, err := os.ReadDir(UploadFolder)
	if err != nil {
		log.Printf("Could not read uploads folder: %v\n", err)
		return
	}

	var mostRecentCSV string
	var mostRecentTime int64 = 0

	for _, file := range files {
		if file.IsDir() || !strings.HasSuffix(strings.ToLower(file.Name()), ".csv") {
			continue
		}

		// Extract timestamp from filename (format: timestamp_filename.csv)
		parts := strings.SplitN(file.Name(), "_", 2)
		if len(parts) < 2 {
			continue
		}

		timestamp, err := strconv.ParseInt(parts[0], 10, 64)
		if err != nil {
			continue
		}

		if timestamp > mostRecentTime {
			mostRecentTime = timestamp
			mostRecentCSV = file.Name()
		}
	}

	if mostRecentCSV != "" {
		filepath := filepath.Join(UploadFolder, mostRecentCSV)
		info, err := processCSVFile(filepath)
		if err != nil {
			log.Printf("Could not load recent CSV %s: %v\n", mostRecentCSV, err)
			return
		}

		fileInfo, _ := os.Stat(filepath)
		lastUploadedFile = &FileInfo{
			Filename:   mostRecentCSV,
			Size:       fileInfo.Size(),
			Type:       ".csv",
			UploadedAt: time.Unix(mostRecentTime, 0),
			Info: map[string]interface{}{
				"csv_info": info,
			},
			ProcessedAt: time.Now(),
		}
		lastCSVData = info
		log.Printf("📊 Auto-loaded most recent CSV: %s (%d rows, %d columns)\n",
			mostRecentCSV, info.Shape[0], info.Shape[1])
	}
}

func processFile(file *multipart.FileHeader, filepath string) (map[string]interface{}, error) {
	ext := strings.ToLower(filepath[len(filepath)-4:])

	result := make(map[string]interface{})

	switch ext {
	case ".csv":
		info, err := processCSVFile(filepath)
		if err != nil {
			return nil, err
		}
		infoJSON, _ := json.Marshal(info)
		json.Unmarshal(infoJSON, &result)
	case ".txt":
		content, err := os.ReadFile(filepath)
		if err != nil {
			return nil, err
		}
		result["content"] = string(content)
		result["size"] = len(content)
	default:
		result["message"] = "File uploaded but processing not implemented for this type"
	}

	return result, nil
}
