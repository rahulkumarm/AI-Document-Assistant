#!/bin/bash

echo "🚀 AI Document Assistant - Deployment Helper"
echo "============================================"

# Check if git remote exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "❌ No git remote 'origin' found!"
    echo ""
    echo "Please create a GitHub repository first:"
    echo "1. Go to https://github.com"
    echo "2. Create a new repository named 'ai-document-assistant'"
    echo "3. Run: git remote add origin https://github.com/YOUR_USERNAME/ai-document-assistant.git"
    echo "4. Then run this script again"
    exit 1
fi

echo "📦 Pushing code to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo "✅ Code pushed successfully!"
    echo ""
    echo "🎯 Next Steps for Railway + Vercel Deployment:"
    echo ""
    echo "BACKEND (Railway):"
    echo "1. Go to https://railway.app"
    echo "2. Sign up/Login with GitHub"
    echo "3. Click 'New Project' → 'Deploy from GitHub repo'"
    echo "4. Select your 'ai-document-assistant' repository"
    echo "5. Add environment variables:"
    echo "   DEBUG=false"
    echo "   HOST=0.0.0.0"
    echo "   PORT=8000"
    echo "   CHROMA_DB_PATH=/app/chroma_db"
    echo "   COLLECTION_NAME=documents"
    echo "   EMBEDDING_MODEL=BAAI/bge-base-en"
    echo ""
    echo "FRONTEND (Vercel):"
    echo "1. Go to https://vercel.com"
    echo "2. Sign up/Login with GitHub"
    echo "3. Click 'New Project'"
    echo "4. Import your 'ai-document-assistant' repository"
    echo "5. Set Root Directory to: frontend"
    echo "6. Framework Preset: Vite"
    echo "7. Add environment variable:"
    echo "   VITE_API_URL=https://your-railway-app.railway.app"
    echo ""
    echo "🔗 Don't forget to update CORS in main_enhanced.py with your Vercel URL!"
    echo ""
else
    echo "❌ Failed to push to GitHub"
    echo "Please check your git remote and try again"
    exit 1
fi 