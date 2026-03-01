#!/bin/bash

# ╔══════════════════════════════════════════════════════════════════╗
# ║         Setup OpenAI API Backend                                 ║
# ║         Simplest solution - just need API key                    ║
# ╚══════════════════════════════════════════════════════════════════╝

cd "$(dirname "$0")/.."

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         🔑 OpenAI API Setup                                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "This is the EASIEST way to get AI text processing working."
echo ""
echo "Pros:"
echo "  ✅ No model downloads"
echo "  ✅ No compilation"
echo "  ✅ Works on any Mac"
echo "  ✅ Best quality results"
echo ""
echo "Cons:"
echo "  ⚠️  Requires internet"
echo "  ⚠️  Uses OpenAI API (small cost, ~$0.002 per transcription)"
echo ""

# Check if already set
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✅ OPENAI_API_KEY is already set!"
    echo ""
    read -n 1 -s -r -p "Press any key to test..."
    
    # Test
    source backend/venv/bin/activate 2>/dev/null || conda activate qwen 2>/dev/null || true
    python -c "
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
print('✅ OpenAI API is working!')
print('Testing with a simple request...')
response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[{'role': 'user', 'content': 'Say hello'}],
    max_tokens=10
)
print(f\"Response: {response.choices[0].message.content}\")
"
    echo ""
    read -n 1 -s -r -p "Press any key to close..."
    exit 0
fi

echo "Step 1: Get API Key"
echo ""
echo "1. Go to: https://platform.openai.com/api-keys"
echo "2. Create an account (if needed)"
echo "3. Click 'Create new secret key'"
echo "4. Copy the key (starts with 'sk-')"
echo ""

read -p "Paste your OpenAI API key: " api_key

if [ -z "$api_key" ]; then
    echo "❌ No key provided"
    exit 1
fi

# Validate key format
if [[ ! "$api_key" =~ ^sk-[a-zA-Z0-9]{20,}$ ]]; then
    echo "❌ Invalid key format. Should start with 'sk-'"
    exit 1
fi

echo ""
echo "Step 2: Installing openai package..."

# Try to activate environment
if [ -d "backend/venv" ]; then
    source backend/venv/bin/activate
elif command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate qwen 2>/dev/null || conda activate base
fi

pip install -q openai

echo ""
echo "Step 3: Testing API key..."

export OPENAI_API_KEY="$api_key"

python -c "
from openai import OpenAI
import os

try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': 'Hello'}],
        max_tokens=5
    )
    print('✅ API key is valid!')
    print(f\"Response: {response.choices[0].message.content}\")
except Exception as e:
    print(f'❌ Error: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ API test failed. Please check your key."
    exit 1
fi

echo ""
echo "Step 4: Saving API key..."

# Add to shell profile
SHELL_PROFILE=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_PROFILE="$HOME/.zshrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_PROFILE="$HOME/.bash_profile"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_PROFILE="$HOME/.bashrc"
fi

if [ -n "$SHELL_PROFILE" ]; then
    # Check if already exists
    if grep -q "OPENAI_API_KEY" "$SHELL_PROFILE"; then
        echo "⚠️  OPENAI_API_KEY already exists in $SHELL_PROFILE"
        echo "    Updating..."
        sed -i.bak '/OPENAI_API_KEY/d' "$SHELL_PROFILE"
    fi
    
    echo "" >> "$SHELL_PROFILE"
    echo "# Qwen3-ASR Pro OpenAI API Key" >> "$SHELL_PROFILE"
    echo "export OPENAI_API_KEY='$api_key'" >> "$SHELL_PROFILE"
    echo "✅ Saved to $SHELL_PROFILE"
else
    echo "⚠️  Could not find shell profile"
    echo "   Please manually add this to your shell profile:"
    echo "   export OPENAI_API_KEY='$api_key'"
fi

# Also save to .env file for current session
echo "OPENAI_API_KEY=$api_key" > .env

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ✅ Setup Complete!                            ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  Usage:                                                    ║"
echo "║    1. Restart Terminal or run:                             ║"
echo "║       source $SHELL_PROFILE"
echo "║                                                            ║"
echo "║    2. Launch app:                                          ║"
echo "║       ./scripts/launch.command                             ║"
echo "║                                                            ║"
echo "║    3. Enable "🤖 AI Text Refiner" in sidebar                ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

read -n 1 -s -r -p "Press any key to close..."
