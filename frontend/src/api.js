// src/api.js
// Centralized API utility for frontend-backend communication

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5000';

// File upload
export async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('Upload failed');
  return res.json();
}

// Chat endpoint (compatible with both Python and Go backends)
export async function sendChatMessage(message) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ error: 'Chat request failed' }));
    throw new Error(errorData.error || 'Chat failed');
  }
  return res.json();
}

// Profile update (placeholder)
export async function updateProfile(profileData) {
  const res = await fetch(`${API_BASE}/profile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(profileData),
  });
  if (!res.ok) throw new Error('Profile update failed');
  return res.json();
}

// Generate chart
export async function generateChart(chartData) {
  const res = await fetch(`${API_BASE}/api/generate-chart`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(chartData),
  });
  if (!res.ok) throw new Error('Chart generation failed');
  return res.json();
}

// Health check
export async function healthCheck() {
  const res = await fetch(`${API_BASE}/`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}
