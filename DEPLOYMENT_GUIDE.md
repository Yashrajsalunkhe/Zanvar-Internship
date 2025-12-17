# Deployment Guide

## Prerequisites
- Node.js 18+ installed
- Go 1.20+ installed
- A server or hosting platform (VPS, AWS, GCP, etc.)
- Domain name (optional but recommended)
- Valid Google Gemini API key

## Step 1: Prepare the Backend

### Build the Go Backend
```bash
cd backend
go build -o server main.go analyzer.go
```

### Configure Production Environment
1. Copy `.env.production.example` to `.env.production`
2. Update the following variables:
   - `GEMINI_API_KEY`: Your production Gemini API key
   - `PORT`: The port for the backend (default: 5000)
   - `ALLOWED_ORIGINS`: Your frontend domain (e.g., https://myapp.com)

### Update CORS in main.go (if needed)
Edit `main.go` and update the CORS configuration:
```go
config := cors.DefaultConfig()
config.AllowOrigins = []string{"https://your-frontend-domain.com"}
config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
config.AllowHeaders = []string{"Origin", "Content-Type", "Accept"}
```

### Run the Backend
```bash
GIN_MODE=release ./server
```

Or use systemd/pm2 for process management:
```bash
# Using systemd
sudo systemctl start zanvar-backend

# Using pm2 (requires pm2 installed)
pm2 start ./server --name zanvar-backend
```

## Step 2: Prepare the Frontend

### Update API URL
Edit `frontend/.env.production`:
```
VITE_API_URL=https://your-backend-domain.com:5000
```

Or if you're using the same domain with a reverse proxy:
```
VITE_API_URL=https://your-domain.com/api
```

### Build the Frontend
```bash
cd frontend
npm install
npm run build
```

This creates a `dist/` folder with production-ready static files.

### Deploy Frontend Static Files

#### Option 1: Nginx
```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /path/to/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

#### Option 2: Apache
```apache
<VirtualHost *:80>
    ServerName your-domain.com
    DocumentRoot /path/to/frontend/dist

    <Directory /path/to/frontend/dist>
        Options Indexes FollowSymLinks
        AllowOverride All
        Require all granted
        
        # Enable React Router
        RewriteEngine On
        RewriteBase /
        RewriteRule ^index\.html$ - [L]
        RewriteCond %{REQUEST_FILENAME} !-f
        RewriteCond %{REQUEST_FILENAME} !-d
        RewriteRule . /index.html [L]
    </Directory>

    ProxyPass /api http://localhost:5000
    ProxyPassReverse /api http://localhost:5000
</VirtualHost>
```

#### Option 3: Netlify/Vercel
1. Push code to GitHub
2. Connect repository to Netlify/Vercel
3. Set build command: `npm run build`
4. Set publish directory: `dist`
5. Add environment variable: `VITE_API_URL`

## Step 3: Configure SSL (Recommended)

### Using Let's Encrypt with Certbot
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal (already configured by certbot)
sudo certbot renew --dry-run
```

## Step 4: Process Management

### Backend with systemd
Create `/etc/systemd/system/zanvar-backend.service`:
```ini
[Unit]
Description=Zanvar Backend Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/backend
Environment="GIN_MODE=release"
EnvironmentFile=/path/to/backend/.env.production
ExecStart=/path/to/backend/server
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable zanvar-backend
sudo systemctl start zanvar-backend
sudo systemctl status zanvar-backend
```

### Alternative: Docker Deployment

#### Backend Dockerfile
Create `backend/Dockerfile`:
```dockerfile
FROM golang:1.20-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o server .

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/server .
COPY --from=builder /app/.env.production .env
EXPOSE 5000
CMD ["./server"]
```

#### Frontend Dockerfile
Create `frontend/Dockerfile`:
```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### Docker Compose
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "5000:5000"
    environment:
      - GIN_MODE=release
    env_file:
      - ./backend/.env.production
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: unless-stopped
```

Run:
```bash
docker-compose up -d
```

## Step 5: Monitoring & Logging

### Backend Logs
```bash
# View systemd logs
sudo journalctl -u zanvar-backend -f

# View Docker logs
docker-compose logs -f backend
```

### Frontend Access Logs
```bash
# Nginx
sudo tail -f /var/log/nginx/access.log

# Docker
docker-compose logs -f frontend
```

## Step 6: Performance Optimization

### Frontend
- Gzip/Brotli compression enabled in nginx/Apache
- CDN for static assets
- Image optimization
- Code splitting (already handled by Vite)

### Backend
- Enable Go profiling for production insights
- Use connection pooling
- Implement caching for frequent requests
- Rate limiting to prevent API abuse

## Troubleshooting

### Backend Issues
1. Check logs: `sudo journalctl -u zanvar-backend`
2. Verify .env file has correct API key
3. Check firewall: `sudo ufw status`
4. Verify port is open: `netstat -tuln | grep 5000`

### Frontend Issues
1. Check nginx/Apache error logs
2. Verify API URL in .env.production
3. Check browser console for CORS errors
4. Ensure build was successful: `ls frontend/dist`

### API Quota Issues
- Monitor Gemini API usage in Google Cloud Console
- Implement rate limiting
- Add caching for common queries
- Consider upgrading API plan

## Security Checklist
- [ ] HTTPS enabled with valid SSL certificate
- [ ] Environment variables secured (not in git)
- [ ] CORS restricted to production domain
- [ ] Firewall configured (only necessary ports open)
- [ ] API key rotated regularly
- [ ] Regular security updates applied
- [ ] File upload size limits enforced
- [ ] Input validation on all endpoints
- [ ] Rate limiting implemented

## Maintenance

### Regular Tasks
- Monitor API usage and costs
- Review logs for errors
- Update dependencies monthly
- Backup uploaded files and generated charts
- Review and rotate API keys quarterly

### Updates
```bash
# Backend
cd backend
go get -u ./...
go mod tidy
go build -o server

# Frontend
cd frontend
npm update
npm audit fix
npm run build
```

## Support
For issues or questions, check the logs first, then review the troubleshooting section.
