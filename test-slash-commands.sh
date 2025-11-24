#!/bin/bash

# Test Script: Verify Slash Command Setup
# This script validates the multi-agent slash command configuration

set -e

PROJECT_DIR="/Users/nitishgautam/Code/prototype/ground-truth"
cd "$PROJECT_DIR"

echo "🧪 Testing Slash Command Setup for Ground-Truth Project"
echo "========================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

# Test function
test_item() {
    local description="$1"
    local test_command="$2"

    echo -n "Testing: $description ... "

    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((FAILED++))
    fi
}

echo "📁 Directory Structure Tests"
echo "----------------------------"
test_item ".claude directory exists" "[ -d .claude ]"
test_item "commands directory exists" "[ -d .claude/commands ]"
test_item "agents directory exists" "[ -d .claude/agents ]"
test_item "commands.json exists" "[ -f .claude/commands.json ]"
echo ""

echo "⚡ Slash Command Files"
echo "----------------------"
test_item "/orchestrate command" "[ -f .claude/commands/orchestrate.md ]"
test_item "/frontend command" "[ -f .claude/commands/frontend.md ]"
test_item "/fastapi command" "[ -f .claude/commands/fastapi.md ]"
test_item "/database command" "[ -f .claude/commands/database.md ]"
test_item "/devops command" "[ -f .claude/commands/devops.md ]"
test_item "/security command" "[ -f .claude/commands/security.md ]"
test_item "/qa command" "[ -f .claude/commands/qa.md ]"
test_item "/review command" "[ -f .claude/commands/review.md ]"
test_item "/design-review command" "[ -f .claude/commands/design-review.md ]"
test_item "/django command" "[ -f .claude/commands/django.md ]"
echo ""

echo "🤖 Agent Configuration Files"
echo "-----------------------------"
test_item "master-orchestrator agent" "[ -f .claude/agents/master-orchestrator.md ]"
test_item "frontend-react-expert agent" "[ -f .claude/agents/frontend-react-expert.md ]"
test_item "backend-fastapi-expert agent" "[ -f .claude/agents/backend-fastapi-expert.md ]"
test_item "database-designer agent" "[ -f .claude/agents/database-designer.md ]"
test_item "devops-engineer agent" "[ -f .claude/agents/devops-engineer.md ]"
test_item "security-architect agent" "[ -f .claude/agents/security-architect.md ]"
test_item "qa-automation-expert agent" "[ -f .claude/agents/qa-automation-expert.md ]"
test_item "code-reviewer agent" "[ -f .claude/agents/code-reviewer.md ]"
test_item "design-reviewer agent" "[ -f .claude/agents/design-reviewer.md ]"
test_item "backend-django-expert agent" "[ -f .claude/agents/backend-django-expert.md ]"
echo ""

echo "📋 Configuration Validation"
echo "---------------------------"
test_item "commands.json is valid JSON" "jq empty .claude/commands.json"
test_item "CLAUDE.md updated with slash commands" "grep -q 'Slash Commands - Multi-Agent Development System' CLAUDE.md"
echo ""

echo "📊 Test Summary"
echo "==============="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed! Slash command system is ready to use.${NC}"
    echo ""
    echo "Next Steps:"
    echo "1. Open this project in Claude Code"
    echo "2. Type '/' to see all available commands"
    echo "3. Try: /orchestrate to start architectural planning"
    echo "4. Try: /fastapi for backend development help"
    echo "5. Try: /database for PostgreSQL schema design"
    echo ""
else
    echo -e "${RED}⚠️  Some tests failed. Please review the setup.${NC}"
    exit 1
fi
