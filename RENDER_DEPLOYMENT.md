# Render Deployment Guide - Quick Setup

Complete guide to deploy Zanvar Data Insights with **Frontend on Vercel** and **Backend on Render**.

---

## Part 1: Deploy Backend to Render

### Step 1: Sign Up & Connect GitHub
1. Go to https://render.com and sign up (free)
2. Click **Dashboard** → **New** → **Web Service**
3. Click **Connect GitHub** and authorize Render
4. Select your `Zanvar-Internship` repository

### Step 2: Configure Backend Service
Fill in the following settings:

**Basic Settings:**
- **Name**: `zanvar-backend` (or your choice)
- **Region**: Oregon (US West) or closest to you
- **Branch**: `main`
- **Root Directory**: `backend`

**Build Settings:**
- **Environment**: `Go`
- **Build Command**: 
  ```
  go build -o server main.go analyzer.go
  ```
- **Start Command**: 
  ```
  ./server
  ```

**Instance Type:**
- Select **Free** (spins down after 15 min inactivity)

### Step 3: Add Environment Variables
Click **Advanced** → **Add Environment Variable** and add:

| Key | Value | Notes |
|-----|-------|-------|
| `PORT` | `5000` | Port for backend |
| `GIN_MODE` | `release` | Production mode |
| `GEMINI_API_KEY` | `your-api-key-here` | Get from Google AI Studio |
| `ALLOWED_ORIGINS` | `https://zanvar-data-insights.vercel.app` | Update with your actual Vercel URL |

**Important**: Replace the Vercel URL with your actual one after deploying frontend.

### Step 4: Create Web Service
1. Click **Create Web Service**
2. Wait 2-3 minutes for deployment
3. You'll get a URL like: `https://zanvar-backend.onrender.com`
4. **Copy this URL** - you'll need it for Vercel!

### Step 5: Verify Backend is Running
Visit: `https://your-backend-url.onrender.com/`

You should see a response (even if it's an error page, it means the server is running).

---

## Part 2: Deploy Frontend to Vercel

### Step 1: Update Environment Variable
Before deploying to Vercel, update the backend URL:

Edit `frontend/.env.production`:
```bash
VITE_API_BASE=https://your-backend-url.onrender.com
```

**Replace** `your-backend-url.onrender.com` with your actual Render URL from Part 1.

### Step 2: Deploy to Vercel

#### Option A: Vercel CLI (Recommended)
```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy from project root
vercel

# Follow prompts:
# - Set up and deploy? Yes
# - Which scope? (Select your account)
# - Link to existing project? No
# - Project name? zanvar-data-insights
# - In which directory is your code located? ./
# - Want to override settings? No

# Deploy to production
vercel --prod
```

#### Option B: Vercel Dashboard (GitHub Integration)
1. Go to https://vercel.com/new
2. Import your Git repository
3. Configure:
   - **Framework Preset**: Other
   - **Root Directory**: `./`
   - **Build Command**: `cd frontend && npm install && npm run build`
   - **Output Directory**: `frontend/dist`
   - **Install Command**: `cd frontend && npm install`
4. Add Environment Variable:
   - **Key**: `VITE_API_BASE`
   - **Value**: `https://your-backend-url.onrender.com`
5. Click **Deploy**

### Step 3: Get Your Vercel URL
After deployment, you'll get a URL like:
```
https://zanvar-data-insights.vercel.app
```

**Copy this URL** - you need to update the backend!

---

## Part 3: Update Backend CORS

### Step 1: Update Render Environment Variable
1. Go to your Render dashboard
2. Select your `zanvar-backend` service
3. Click **Environment** on the left
4. Find `ALLOWED_ORIGINS` and update it to your Vercel URL:
   ```
   https://zanvar-data-insights.vercel.app
   ```
5. Click **Save Changes** - Render will auto-redeploy

### Step 2: Update CORS in Code (Optional but Recommended)
Edit `backend/main.go` to hardcode your Vercel URL:

Find the CORS configuration section and update:
```go
config := cors.DefaultConfig()
config.AllowOrigins = []string{
    "https://zanvar-data-insights.vercel.app",  // Your actual Vercel URL
}
config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
config.AllowHeaders = []string{"Origin", "Content-Type", "Accept"}
config.AllowCredentials = true
```

Commit and push:
```bash
git add backend/main.go
git commit -m "Update CORS for Vercel deployment"
git push origin main
```

Render will automatically redeploy within 1-2 minutes.

---

## Part 4: Test Your Deployment

### Visit Your App
Go to your Vercel URL: `https://zanvar-data-insights.vercel.app`

### Test Checklist:
- [ ] **Homepage loads** - Should see the landing page
- [ ] **Upload page** - Try uploading a CSV file
- [ ] **File upload works** - Check if file uploads successfully
- [ ] **Chat page** - Ask a question about your data
- [ ] **AI responds** - Verify Gemini API integration
- [ ] **Charts display** - Check if visualizations render
- [ ] **No CORS errors** - Open browser console (F12), check for errors

---

## Troubleshooting

### Issue: CORS Error in Browser Console
**Error**: `Access to fetch at 'https://...' from origin 'https://...' has been blocked by CORS policy`

**Fix**:
1. Verify `ALLOWED_ORIGINS` in Render dashboard matches your Vercel URL exactly
2. Check `backend/main.go` CORS configuration
3. Make sure there's no trailing slash in URLs
4. Redeploy backend after changes

### Issue: Backend Not Responding / 500 Error
**Fix**:
1. Check Render logs: Dashboard → Your service → **Logs**
2. Verify `GEMINI_API_KEY` is set correctly
3. Check if backend is active (free tier spins down after 15 min)
4. Visit backend URL to wake it up

### Issue: "Failed to fetch" or Network Error
**Fix**:
1. Verify `VITE_API_BASE` in Vercel environment variables
2. Check if backend URL is correct (should be HTTPS)
3. Ensure backend is running (check Render dashboard)
4. Test backend directly: `https://your-backend.onrender.com/`

### Issue: Upload Fails
**Fix**:
1. Check file size (Render free tier has memory limits)
2. Verify `uploads/` directory permissions
3. Check Render logs for errors
4. Ensure `GEMINI_API_KEY` is valid

### Issue: Render Service Keeps Spinning Down
**Note**: Free tier spins down after 15 minutes of inactivity. First request after spin-down may take 30-60 seconds (cold start).

**Solutions**:
- Upgrade to paid plan ($7/month) for always-on service
- Use a uptime monitoring service (like UptimeRobot) to ping every 14 minutes
- Accept cold starts as part of free tier

---

## Render Free Tier Limits

- **750 hours/month** of runtime
- **512 MB RAM** per service
- **Spins down** after 15 min inactivity
- **No credit card** required
- **Custom domains** supported (free SSL)

---

## Updating Your App

### Update Frontend:
```bash
# Make changes to frontend code
git add .
git commit -m "Update frontend"
git push origin main
# Vercel auto-deploys
```

### Update Backend:
```bash
# Make changes to backend code
git add .
git commit -m "Update backend"
git push origin main
# Render auto-deploys (check dashboard for progress)
```

### Update Environment Variables:
**Vercel**: Dashboard → Project → Settings → Environment Variables → Edit → Save → Redeploy

**Render**: Dashboard → Service → Environment → Edit → Save Changes (auto-redeploys)

---

## Custom Domain (Optional)

### Add Custom Domain to Vercel:
1. Vercel Dashboard → Your project → **Settings** → **Domains**
2. Add domain: `app.yourdomain.com`
3. Configure DNS records as shown

### Add Custom Domain to Render:
1. Render Dashboard → Your service → **Settings**
2. Scroll to **Custom Domain**
3. Add domain: `api.yourdomain.com`
4. Update DNS records (CNAME)

**Don't forget** to update:
- `VITE_API_BASE` in Vercel (to new backend domain)
- `ALLOWED_ORIGINS` in Render (to new frontend domain)

---

## Monitoring

### Render Dashboard:
- **Logs**: Real-time backend logs
- **Metrics**: CPU, Memory, Request count
- **Events**: Deployment history
- **Shell**: Access terminal (paid plans)

### Vercel Dashboard:
- **Deployments**: Build logs and status
- **Analytics**: Visitor stats (requires upgrade)
- **Speed Insights**: Performance metrics

---

## Cost Summary

**Current Setup (Free Tier)**:
- **Vercel**: FREE forever
  - 100 GB bandwidth/month
  - Unlimited deployments
  - Custom domain + SSL

- **Render**: FREE
  - 750 hours/month (≈ 1 service running 24/7)
  - 512 MB RAM
  - ⚠️ Cold starts after inactivity

**Total Cost**: $0/month (perfect for portfolio/demo projects)

**To Upgrade** (optional):
- **Render Starter Plan**: $7/month
  - Always-on (no cold starts)
  - 512 MB RAM
  - Better performance

---

## Quick Command Reference

```bash
# Deploy frontend to Vercel
vercel --prod

# View Render logs (if using Render CLI)
render logs -s zanvar-backend

# Check backend health
curl https://your-backend-url.onrender.com/

# View Vercel deployment
vercel inspect

# Git workflow
git add .
git commit -m "Your changes"
git push origin main
# Both platforms auto-deploy!
```

---

## Next Steps

1. ✅ Backend deployed on Render
2. ✅ Frontend deployed on Vercel
3. ✅ Environment variables configured
4. ✅ CORS updated
5. 🎉 **Your app is live!**

Share your URLs:
- **Frontend**: `https://your-app.vercel.app`
- **Backend**: `https://your-backend.onrender.com`

---

## Support & Resources

- **Render Docs**: https://render.com/docs
- **Vercel Docs**: https://vercel.com/docs
- **Render Community**: https://community.render.com/
- **Vercel Discord**: https://vercel.com/discord

Good luck with your deployment! 🚀
