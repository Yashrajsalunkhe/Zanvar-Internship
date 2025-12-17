package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
)

// DataAnalyzer provides statistical analysis capabilities
type DataAnalyzer struct {
	Data    [][]string
	Headers []string
}

// NewDataAnalyzer creates a new data analyzer from CSV file
func NewDataAnalyzer(filepath string) (*DataAnalyzer, error) {
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

	if len(records) < 2 {
		return nil, fmt.Errorf("insufficient data in CSV")
	}

	return &DataAnalyzer{
		Data:    records[1:],
		Headers: records[0],
	}, nil
}

// GetColumn extracts a specific column by name
func (da *DataAnalyzer) GetColumn(columnName string) ([]string, error) {
	colIndex := -1
	for i, header := range da.Headers {
		if header == columnName {
			colIndex = i
			break
		}
	}

	if colIndex == -1 {
		return nil, fmt.Errorf("column '%s' not found", columnName)
	}

	column := make([]string, len(da.Data))
	for i, row := range da.Data {
		if colIndex < len(row) {
			column[i] = row[colIndex]
		}
	}

	return column, nil
}

// GetNumericColumn extracts a numeric column
func (da *DataAnalyzer) GetNumericColumn(columnName string) ([]float64, error) {
	strColumn, err := da.GetColumn(columnName)
	if err != nil {
		return nil, err
	}

	numColumn := make([]float64, 0, len(strColumn))
	for _, val := range strColumn {
		if val == "" {
			continue
		}
		num, err := strconv.ParseFloat(val, 64)
		if err != nil {
			continue
		}
		numColumn = append(numColumn, num)
	}

	if len(numColumn) == 0 {
		return nil, fmt.Errorf("no numeric values found in column '%s'", columnName)
	}

	return numColumn, nil
}

// Statistics holds statistical measures
type Statistics struct {
	Mean   float64 `json:"mean"`
	Median float64 `json:"median"`
	Mode   float64 `json:"mode"`
	StdDev float64 `json:"std_dev"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	Count  int     `json:"count"`
}

// CalculateStatistics computes statistics for a numeric column
func (da *DataAnalyzer) CalculateStatistics(columnName string) (*Statistics, error) {
	data, err := da.GetNumericColumn(columnName)
	if err != nil {
		return nil, err
	}

	if len(data) == 0 {
		return nil, fmt.Errorf("no data to analyze")
	}

	stats := &Statistics{
		Count: len(data),
	}

	// Calculate mean
	sum := 0.0
	for _, val := range data {
		sum += val
	}
	stats.Mean = sum / float64(len(data))

	// Calculate min and max
	stats.Min = data[0]
	stats.Max = data[0]
	for _, val := range data {
		if val < stats.Min {
			stats.Min = val
		}
		if val > stats.Max {
			stats.Max = val
		}
	}

	// Calculate median
	sortedData := make([]float64, len(data))
	copy(sortedData, data)
	sort.Float64s(sortedData)

	mid := len(sortedData) / 2
	if len(sortedData)%2 == 0 {
		stats.Median = (sortedData[mid-1] + sortedData[mid]) / 2
	} else {
		stats.Median = sortedData[mid]
	}

	// Calculate standard deviation
	variance := 0.0
	for _, val := range data {
		variance += math.Pow(val-stats.Mean, 2)
	}
	variance /= float64(len(data))
	stats.StdDev = math.Sqrt(variance)

	// Calculate mode (most frequent value)
	frequency := make(map[float64]int)
	maxFreq := 0
	for _, val := range data {
		frequency[val]++
		if frequency[val] > maxFreq {
			maxFreq = frequency[val]
			stats.Mode = val
		}
	}

	return stats, nil
}

// GetColumnNames returns all column names
func (da *DataAnalyzer) GetColumnNames() []string {
	return da.Headers
}

// GetRowCount returns the number of rows
func (da *DataAnalyzer) GetRowCount() int {
	return len(da.Data)
}

// GetColumnCount returns the number of columns
func (da *DataAnalyzer) GetColumnCount() int {
	return len(da.Headers)
}

// ValueCounts counts unique values in a column
func (da *DataAnalyzer) ValueCounts(columnName string) (map[string]int, error) {
	column, err := da.GetColumn(columnName)
	if err != nil {
		return nil, err
	}

	counts := make(map[string]int)
	for _, val := range column {
		if val != "" {
			counts[val]++
		}
	}

	return counts, nil
}

// Correlation calculates correlation between two numeric columns
func (da *DataAnalyzer) Correlation(col1Name, col2Name string) (float64, error) {
	data1, err := da.GetNumericColumn(col1Name)
	if err != nil {
		return 0, err
	}

	data2, err := da.GetNumericColumn(col2Name)
	if err != nil {
		return 0, err
	}

	if len(data1) != len(data2) {
		return 0, fmt.Errorf("columns have different lengths")
	}

	if len(data1) == 0 {
		return 0, fmt.Errorf("no data to correlate")
	}

	// Calculate means
	mean1 := 0.0
	mean2 := 0.0
	for i := range data1 {
		mean1 += data1[i]
		mean2 += data2[i]
	}
	mean1 /= float64(len(data1))
	mean2 /= float64(len(data2))

	// Calculate correlation
	numerator := 0.0
	denominator1 := 0.0
	denominator2 := 0.0

	for i := range data1 {
		diff1 := data1[i] - mean1
		diff2 := data2[i] - mean2
		numerator += diff1 * diff2
		denominator1 += diff1 * diff1
		denominator2 += diff2 * diff2
	}

	if denominator1 == 0 || denominator2 == 0 {
		return 0, nil
	}

	correlation := numerator / math.Sqrt(denominator1*denominator2)
	return correlation, nil
}
