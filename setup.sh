#!/bin/bash
# Cortex v3.0 Setup Script
# Run this after installing Ollama

set -e

echo "=============================================="
echo "CORTEX v3.0 SETUP"
echo "=============================================="

# Check Ollama
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "❌ Ollama not installed!"
    echo "   Run: curl -fsSL https://ollama.com/install.sh | sh"
    echo "   Then run this script again."
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo ""
    echo "⚠️ Ollama not running. Starting..."
    ollama serve &
    sleep 3
fi

echo ""
echo "📥 Pulling models..."

# Essential models
echo ""
echo "1. Pulling nomic-embed-text (embeddings, ~300MB)..."
ollama pull nomic-embed-text

echo ""
echo "2. Pulling qwen2.5-coder:7b (fast queries, ~5GB)..."
ollama pull qwen2.5-coder:7b

# Ask about large model
echo ""
read -p "Pull qwen2.5-coder:32b (18GB VRAM, much more capable)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "3. Pulling qwen2.5-coder:32b (~18GB)..."
    ollama pull qwen2.5-coder:32b
fi

echo ""
echo "=============================================="
echo "✅ SETUP COMPLETE"
echo "=============================================="
echo ""
echo "Quick commands:"
echo "  cortex status    - Show system status"
echo "  cortex test      - Run system tests"
echo "  cortex stats     - Show statistics"
echo "  cortex analyze . - Analyze current project"
echo ""
echo "The hooks are already configured."
echo "Start a new Claude Code session to see Cortex v3.0 in action!"
