# Fix 404 Error - Deployment Checklist

## Current Issue
Your frontend is deployed on Vercel, but when you click "Upload File", you get a 404 error because:
- The frontend is trying to call `https://your-backend-url.com/api/upload` (placeholder URL)
- The backend is not deployed yet, or the environment variable is not configured

## Solution Steps

### ✅ Step 1: Deploy Backend to Render

1. **Go to** https://render.com and sign in
2. **Click** "New" → "Web Service"
3. **Connect** your GitHub repository: `Zanvar-Internship`
4. **Configure** the service:
   - **Name**: `zanvar-backend` (or any name you prefer)
   - **Region**: Oregon (US West) or closest to you
   - **Branch**: `main`
   - **Root Directory**: `backend`
   - **Environment**: `Go`
   - **Build Command**: 
     ```
     go build -o server main.go analyzer.go
     ```
   - **Start Command**: 
     ```
     ./server
     ```
   - **Instance Type**: Free

5. **Add Environment Variables** (click "Advanced" button):
   - `PORT` = `5000`
   - `GIN_MODE` = `release`
   - `GOOGLE_API_KEY` = `your-google-gemini-api-key` (get from https://aistudio.google.com/app/apikey)
   - `ALLOWED_ORIGINS` = `https://your-vercel-url.vercel.app` (you'll update this after Step 2)

6. **Click** "Create Web Service"
7. **Wait** 2-3 minutes for deployment
8. **Copy** your backend URL (will be like `https://zanvar-backend.onrender.com`)

### ✅ Step 2: Update Vercel Environment Variable

1. **Go to** https://vercel.com/dashboard
2. **Click** on your project (e.g., `zanvar-data-insights`)
3. **Go to** Settings → Environment Variables
4. **Look for** `VITE_API_BASE` or **Add new variable**:
   - **Key**: `VITE_API_BASE`
   - **Value**: `https://your-backend-url.onrender.com` (paste the URL from Step 1)
   - **Environments**: Select all (Production, Preview, Development)
5. **Click** "Save"
6. **Go to** Deployments tab
7. **Click** the 3 dots on the latest deployment → **Redeploy**
8. **Copy** your Vercel URL (e.g., `https://zanvar-data-insights.vercel.app`)

### ✅ Step 3: Update Backend CORS

1. **Go back to** Render Dashboard
2. **Click** on your `zanvar-backend` service
3. **Click** "Environment" in the left sidebar
4. **Find** `ALLOWED_ORIGINS` variable
5. **Update** the value to your Vercel URL from Step 2:
   ```
   https://zanvar-data-insights.vercel.app
   ```
   (use your actual Vercel URL - no trailing slash!)
6. **Click** "Save Changes"
7. Render will automatically redeploy (wait 1-2 minutes)

### ✅ Step 4: Test Your Deployment

1. **Visit** your Vercel URL: `https://your-app.vercel.app`
2. **Navigate** to the Upload page
3. **Try uploading** a CSV file
4. **Check** if it uploads successfully and redirects to chat

### 🔍 Troubleshooting

#### If you still get errors:

**Open Browser Console** (Press F12) and check for errors:

- **CORS Error**: Make sure `ALLOWED_ORIGINS` in Render matches your Vercel URL exactly (no trailing slash)
- **Network Error / Failed to fetch**: 
  - Check if backend URL in Vercel env vars is correct
  - Visit your backend URL directly to see if it's running
  - Free tier Render backends sleep after 15 minutes - visit the URL to wake it up
- **500 Error**: 
  - Check Render logs (Dashboard → Your Service → Logs)
  - Verify `GOOGLE_API_KEY` is set correctly

#### Quick Test Commands

**Test Backend Health** (should return JSON):
```bash
curl https://your-backend-url.onrender.com/
```

**Test Backend Upload Endpoint**:
```bash
curl -X POST https://your-backend-url.onrender.com/api/upload
# Should return: {"error":"No file uploaded"}
```

## Environment Variables Summary

### Render (Backend)
- `PORT` = `5000`
- `GIN_MODE` = `release`
- `GOOGLE_API_KEY` = `your-actual-google-api-key`
- `ALLOWED_ORIGINS` = `https://your-vercel-app.vercel.app`

### Vercel (Frontend)
- `VITE_API_BASE` = `https://your-backend.onrender.com`

## Important Notes

1. **No Trailing Slashes**: Don't add `/` at the end of URLs
2. **HTTPS Only**: Both URLs must use HTTPS in production
3. **Exact Match**: CORS requires exact URL match (including protocol)
4. **Free Tier Sleep**: Render free tier sleeps after 15 minutes of inactivity - first request may be slow
5. **Redeploy Required**: After changing environment variables, you must redeploy (Vercel) or it auto-redeploys (Render)

## Get Your Google API Key

If you don't have a Google Gemini API key:
1. Go to https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and add it to Render environment variables

---

**After completing all steps, your app should work!** 🎉

If you still face issues, check:
- Render logs for backend errors
- Browser console (F12) for frontend errors
- Both environment variables are set correctly
