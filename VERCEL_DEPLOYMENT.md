# Split Deployment Guide - Vercel + Backend Platform

This guide shows you how to deploy the Zanvar Data Insights app using split deployment:
- **Frontend**: Vercel (React + Vite)
- **Backend**: Railway, Render, or Fly.io (Go)

---

## Part 1: Deploy Frontend to Vercel

### Step 1: Prepare Your Repository
The project is already configured with:
- ✅ `vercel.json` - Vercel configuration
- ✅ `.vercelignore` - Excludes backend files
- ✅ `frontend/.env.production` - Production environment template

### Step 2: Deploy to Vercel

#### Option A: Using Vercel CLI
```bash
# Install Vercel CLI globally
npm install -g vercel

# Login to Vercel
vercel login

# Deploy from project root
vercel

# Follow the prompts:
# - Set up and deploy? Yes
# - Which scope? (Select your account)
# - Link to existing project? No
# - Project name? zanvar-data-insights (or your choice)
# - In which directory is your code located? ./
# - Want to override settings? No

# For production deployment
vercel --prod
```

#### Option B: Using Vercel Dashboard
1. Go to https://vercel.com/new
2. Import your Git repository (GitHub/GitLab/Bitbucket)
3. Configure project:
   - **Framework Preset**: Other
   - **Root Directory**: `./` (keep as is)
   - **Build Command**: `cd frontend && npm install && npm run build`
   - **Output Directory**: `frontend/dist`
   - **Install Command**: `cd frontend && npm install`
4. Add Environment Variable:
   - **Key**: `VITE_API_BASE`
   - **Value**: `https://your-backend-url.com` (update after deploying backend)
5. Click **Deploy**

### Step 3: Note Your Vercel URL
After deployment, you'll get a URL like: `https://zanvar-data-insights.vercel.app`

---

## Part 2: Deploy Backend (Choose One Platform)

### Option A: Deploy to Railway (Recommended - Easiest)

#### Why Railway?
- ✅ Native Go support
- ✅ Free tier available ($5 credit/month)
- ✅ Automatic HTTPS
- ✅ Simple environment variables
- ✅ GitHub integration

#### Steps:
1. **Sign up** at https://railway.app
2. **New Project** → **Deploy from GitHub repo**
3. **Select your repository**
4. **Configure**:
   - Railway auto-detects Go projects
   - **Root Directory**: `backend`
   - **Build Command**: `go build -o server main.go analyzer.go`
   - **Start Command**: `./server`
5. **Add Environment Variables**:
   - `GEMINI_API_KEY`: Your Google Gemini API key
   - `PORT`: `5000`
   - `GIN_MODE`: `release`
   - `ALLOWED_ORIGINS`: `https://your-vercel-app.vercel.app`
6. **Deploy** - Railway will provide a URL like `https://zanvar-backend.up.railway.app`

#### Update CORS in backend/main.go:
```go
config := cors.DefaultConfig()
config.AllowOrigins = []string{
    "https://your-vercel-app.vercel.app",  // Your Vercel frontend URL
}
config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
config.AllowHeaders = []string{"Origin", "Content-Type", "Accept"}
config.AllowCredentials = true
```

---

### Option B: Deploy to Render

#### Why Render?
- ✅ Native Go support
- ✅ Free tier available
- ✅ Automatic HTTPS
- ✅ Good for long-running services

#### Steps:
1. **Sign up** at https://render.com
2. **New** → **Web Service**
3. **Connect** your GitHub repository
4. **Configure**:
   - **Name**: `zanvar-backend`
   - **Root Directory**: `backend`
   - **Environment**: `Go`
   - **Build Command**: `go build -o server main.go analyzer.go`
   - **Start Command**: `./server`
   - **Instance Type**: Free
5. **Environment Variables**:
   - `GEMINI_API_KEY`: Your API key
   - `PORT`: `5000`
   - `GIN_MODE`: `release`
   - `ALLOWED_ORIGINS`: `https://your-vercel-app.vercel.app`
6. **Create Web Service**
7. You'll get a URL like: `https://zanvar-backend.onrender.com`

**Note**: Free tier on Render spins down after inactivity (may have cold starts).

---

### Option C: Deploy to Fly.io

#### Why Fly.io?
- ✅ Native Go support
- ✅ Free tier (3 shared VMs)
- ✅ Global edge deployment
- ✅ Great performance

#### Steps:
1. **Install Fly CLI**:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login**:
   ```bash
   fly auth login
   ```

3. **Navigate to backend**:
   ```bash
   cd backend
   ```

4. **Initialize Fly app**:
   ```bash
   fly launch
   # Choose app name: zanvar-backend
   # Choose region: (closest to you)
   # Don't deploy yet: No
   ```

5. **Edit fly.toml** (created automatically):
   ```toml
   app = "zanvar-backend"
   primary_region = "iad"

   [build]
   [build.args]
     GO_VERSION = "1.20"

   [env]
     PORT = "8080"
     GIN_MODE = "release"

   [[services]]
     internal_port = 5000
     protocol = "tcp"

     [[services.ports]]
       port = 80
       handlers = ["http"]
       force_https = true

     [[services.ports]]
       port = 443
       handlers = ["tls", "http"]

   [http_service]
     internal_port = 5000
     force_https = true
     auto_stop_machines = true
     auto_start_machines = true
   ```

6. **Set secrets**:
   ```bash
   fly secrets set GEMINI_API_KEY="your-api-key-here"
   fly secrets set ALLOWED_ORIGINS="https://your-vercel-app.vercel.app"
   ```

7. **Deploy**:
   ```bash
   fly deploy
   ```

8. Your app will be at: `https://zanvar-backend.fly.dev`

---

## Part 3: Connect Frontend to Backend

### Step 1: Update Vercel Environment Variable
1. Go to your Vercel dashboard
2. Select your project → **Settings** → **Environment Variables**
3. Update `VITE_API_BASE`:
   - Railway: `https://zanvar-backend.up.railway.app`
   - Render: `https://zanvar-backend.onrender.com`
   - Fly.io: `https://zanvar-backend.fly.dev`
4. **Redeploy** your Vercel app (or it will auto-deploy on next commit)

### Step 2: Update Backend CORS
Edit `backend/main.go` and update allowed origins:
```go
config.AllowOrigins = []string{
    "https://your-actual-vercel-url.vercel.app",
}
```
Commit and push - your backend platform will auto-redeploy.

---

## Part 4: Verify Deployment

### Test Checklist:
- [ ] Frontend loads at Vercel URL
- [ ] File upload works
- [ ] AI chat responds correctly
- [ ] Charts generate properly
- [ ] No CORS errors in browser console
- [ ] Backend logs show requests (check platform dashboard)

### Common Issues:

**CORS Errors?**
- Check `ALLOWED_ORIGINS` environment variable on backend
- Verify `main.go` CORS configuration matches Vercel URL
- Ensure both HTTP and HTTPS are considered if testing

**Upload/Chat Not Working?**
- Verify `VITE_API_BASE` in Vercel environment variables
- Check backend logs for errors
- Ensure `GEMINI_API_KEY` is set correctly on backend

**Backend Not Starting?**
- Check backend platform logs
- Verify `PORT` environment variable
- Ensure all Go dependencies are in `go.mod`

---

## Cost Breakdown

### Free Tier Limits:

**Vercel** (Frontend):
- 100 GB bandwidth/month
- Unlimited deployments
- Custom domain support
- ✅ Perfect for this project

**Railway** (Backend - Recommended):
- $5 free credit/month (~500 hours)
- 8GB RAM, 8vCPU
- ⚠️ Credit expires monthly

**Render** (Backend):
- 750 hours/month free
- 512MB RAM
- ⚠️ Spins down after 15min inactivity (cold starts)

**Fly.io** (Backend):
- 3 shared-cpu-1x VMs free
- 256MB RAM each
- 3GB persistent volume
- ✅ No cold starts

### Recommended for Zero Cost:
- **Frontend**: Vercel (free forever)
- **Backend**: Fly.io (stays warm, no cold starts)

---

## Monitoring & Maintenance

### Vercel Dashboard:
- View deployment logs
- Monitor bandwidth usage
- Configure custom domains
- Manage environment variables

### Backend Platform Dashboard:
- **Railway**: Metrics, logs, database (if needed)
- **Render**: Logs, metrics, shell access
- **Fly.io**: Metrics, logs, SSH access

---

## Custom Domain Setup (Optional)

### For Vercel (Frontend):
1. Go to your Vercel project → **Settings** → **Domains**
2. Add your custom domain (e.g., `app.yourdomain.com`)
3. Configure DNS records as shown

### For Backend:
- **Railway**: Project Settings → Domains → Add custom domain
- **Render**: Service → Settings → Custom Domain
- **Fly.io**: `fly certs add yourdomain.com`

Don't forget to update `VITE_API_BASE` and `ALLOWED_ORIGINS` after domain changes!

---

## Quick Start Commands

### Deploy Frontend (Vercel CLI):
```bash
vercel --prod
```

### Deploy Backend (Railway):
```bash
# Push to GitHub - Railway auto-deploys
git push origin main
```

### Deploy Backend (Fly.io):
```bash
cd backend
fly deploy
```

---

## Next Steps

1. ✅ Deploy frontend to Vercel
2. ✅ Choose and deploy backend (Railway recommended)
3. ✅ Update environment variables on both platforms
4. ✅ Update CORS configuration
5. ✅ Test the deployed application
6. 🎉 Share your live URL!

---

## Support

**Vercel Issues**: https://vercel.com/docs
**Railway Issues**: https://docs.railway.app
**Render Issues**: https://render.com/docs
**Fly.io Issues**: https://fly.io/docs

Good luck with your deployment! 🚀
