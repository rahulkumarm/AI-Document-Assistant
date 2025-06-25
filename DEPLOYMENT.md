# 🚀 Deployment Guide

This guide covers multiple hosting options for your full-stack AI application.

## 📋 Pre-deployment Checklist

### Backend Requirements
- [ ] FastAPI application (`main_enhanced.py`)
- [ ] Dependencies in `requirements.txt`
- [ ] ChromaDB for vector storage
- [ ] Sentence Transformers for embeddings
- [ ] LLM integration (Ollama/local models)

### Frontend Requirements  
- [ ] React/Vite application in `frontend/` directory
- [ ] Built with `npm run build`
- [ ] Environment variables configured

## 🏗️ Hosting Options

### Option 1: Railway + Vercel (Recommended)

**Backend on Railway:**
1. Push code to GitHub
2. Go to [railway.app](https://railway.app)
3. Create new project from GitHub repo
4. Railway auto-detects Python/FastAPI
5. Add environment variables:
   ```
   DEBUG=False
   HOST=0.0.0.0
   PORT=8000
   CHROMA_DB_PATH=/app/chroma_db
   COLLECTION_NAME=documents
   EMBEDDING_MODEL=BAAI/bge-base-en
   ```
6. Deploy automatically

**Frontend on Vercel:**
```bash
cd frontend
npm run build
npx vercel --prod
```

**Cost**: ~$5-20/month for small-medium usage

---

### Option 2: Render (Full-stack)

**Backend Service:**
1. Go to [render.com](https://render.com)
2. Create "Web Service" from GitHub
3. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main_enhanced:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Add production variables

**Frontend Static Site:**
1. Create "Static Site" from GitHub
2. Configure:
   - **Build Command**: `cd frontend && npm install && npm run build`
   - **Publish Directory**: `frontend/dist`

**Cost**: Free tier available, paid plans start at $7/month

---

### Option 3: DigitalOcean App Platform

1. Create account at [digitalocean.com](https://digitalocean.com)
2. Go to App Platform
3. Connect GitHub repository
4. Configure services:

**Backend Configuration:**
```yaml
services:
- name: backend
  source_dir: /
  github:
    repo: your-username/your-repo
    branch: main
  run_command: uvicorn main_enhanced:app --host 0.0.0.0 --port $PORT
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  envs:
  - key: DEBUG
    value: "False"
  - key: CHROMA_DB_PATH
    value: "/app/chroma_db"
```

**Frontend Configuration:**
```yaml
- name: frontend
  source_dir: /frontend
  github:
    repo: your-username/your-repo
    branch: main
  build_command: npm run build
  output_dir: dist
  environment_slug: node-js
```

**Cost**: $12-25/month for small apps

---

### Option 4: Docker + VPS (Most Control)

**1. Build and Push Images:**
```bash
# Build backend
docker build -t your-app-backend .

# Build frontend  
cd frontend
docker build -t your-app-frontend .

# Push to Docker Hub or registry
docker push your-app-backend
docker push your-app-frontend
```

**2. Deploy on VPS:**
```bash
# On your VPS (DigitalOcean Droplet, AWS EC2, etc.)
docker-compose up -d
```

**Cost**: $5-50/month depending on VPS size

---

### Option 5: AWS/GCP (Enterprise)

**AWS Setup:**
- **Backend**: ECS Fargate or Lambda
- **Frontend**: S3 + CloudFront
- **Database**: RDS or DynamoDB for metadata
- **Vector Store**: Persist ChromaDB to EFS

**GCP Setup:**  
- **Backend**: Cloud Run
- **Frontend**: Firebase Hosting or Cloud Storage
- **Database**: Cloud SQL or Firestore

**Cost**: Pay-per-use, can be $10-100+/month

---

## 🔧 Quick Deploy Commands

### Local Testing
```bash
# Test with Docker Compose
docker-compose up --build

# Access:
# Frontend: http://localhost:3000  
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Railway Deploy
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway link
railway up
```

### Vercel Deploy (Frontend)
```bash
cd frontend
npm install -g vercel
vercel --prod
```

### Render Deploy
```bash
# Just push to GitHub and connect in Render dashboard
git add .
git commit -m "Deploy to production"
git push origin main
```

## ⚙️ Environment Variables

### Backend (.env)
```bash
# Required
DEBUG=False
HOST=0.0.0.0
PORT=8000
CHROMA_DB_PATH=/app/chroma_db
COLLECTION_NAME=documents
EMBEDDING_MODEL=BAAI/bge-base-en

# Optional (for LLM)
MODEL_PATH=./models/llama-2-7b-chat.q4_0.gguf
MODEL_N_CTX=2048
MODEL_TEMPERATURE=0.7

# Security (production)
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=your-domain.com
CORS_ORIGINS=https://your-frontend-domain.com
```

### Frontend (.env)
```bash
VITE_API_URL=https://your-backend-domain.com
VITE_APP_NAME=AI Document Assistant
```

## 📊 Resource Requirements

### Minimum (Development)
- **RAM**: 2GB
- **CPU**: 1 core  
- **Storage**: 5GB
- **Bandwidth**: 100GB/month

### Recommended (Production)
- **RAM**: 4-8GB
- **CPU**: 2-4 cores
- **Storage**: 20GB SSD
- **Bandwidth**: 500GB/month

### Heavy Usage (Enterprise)
- **RAM**: 16GB+
- **CPU**: 8+ cores
- **Storage**: 100GB+ SSD
- **GPU**: Optional for faster embeddings

## 🔒 Security Considerations

### Backend Security
- [ ] Use HTTPS in production
- [ ] Set proper CORS origins
- [ ] Add rate limiting
- [ ] Validate file uploads
- [ ] Sanitize user inputs

### Frontend Security  
- [ ] Environment variables for API URLs
- [ ] Content Security Policy headers
- [ ] Secure authentication (if added)

## 📈 Monitoring & Logging

### Basic Monitoring
```python
# Add to main_enhanced.py
import logging
from fastapi import Request
import time

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.2f}s")
    return response
```

### Production Monitoring
- **Uptime**: UptimeRobot, Pingdom
- **Logs**: Papertrail, LogDNA  
- **Metrics**: Datadog, New Relic
- **Errors**: Sentry

## 🚀 Performance Optimization

### Backend Optimization
```python
# Add caching
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_embeddings(text: str):
    return embedder.encode([text])

# Async operations
import asyncio
async def process_multiple_docs(docs):
    tasks = [process_doc(doc) for doc in docs]
    return await asyncio.gather(*tasks)
```

### Frontend Optimization
```javascript
// Code splitting
const DocumentViewer = lazy(() => import('./DocumentViewer'));

// Caching
import { QueryClient, QueryClientProvider } from 'react-query';
const queryClient = new QueryClient();
```

## 📞 Support & Troubleshooting

### Common Issues

**Backend not starting:**
```bash
# Check logs
docker logs container_name

# Common fixes
pip install --upgrade sentence-transformers
export PYTORCH_ENABLE_MPS_FALLBACK=1  # For Mac M1/M2
```

**Frontend build fails:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
npm run build
```

**CORS errors:**
```python
# Update CORS in main_enhanced.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🎯 Recommended Deployment Flow

For most users, I recommend:

1. **Start with Railway + Vercel** for simplicity
2. **Scale to DigitalOcean** if you need more control  
3. **Move to AWS/GCP** for enterprise needs

This gives you a clear upgrade path as your application grows! 