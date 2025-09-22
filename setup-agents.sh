#!/bin/bash

# Universal Agent Setup for any project in Code/
# Usage: ./setup-agents.sh [project-name] [agent-name]

PROJECT_NAME=${1:-$(basename $(pwd))}
AGENT_NAME=${2:-"design-reviewer"}

echo "🚀 Setting up $AGENT_NAME agent for project: $PROJECT_NAME"

# Check if we're in a project directory
if [ ! -d ".git" ] && [ ! -f "package.json" ] && [ ! -f "index.html" ]; then
    echo "⚠️  This doesn't look like a project directory"
    echo "   Make sure you're in your project root"
    exit 1
fi

# Set the correct agents path
CLAUDE_AGENTS_PATH="/Users/nitishgautam/Code/.claude-agents"

# Set up the specified agent
if [ -x "$CLAUDE_AGENTS_PATH/$AGENT_NAME/scripts/setup.sh" ]; then
    "$CLAUDE_AGENTS_PATH/$AGENT_NAME/scripts/setup.sh"
else
    echo "❌ Agent $AGENT_NAME not found in $CLAUDE_AGENTS_PATH"
    echo "Available agents:"
    ls -1 "$CLAUDE_AGENTS_PATH/"
fi
