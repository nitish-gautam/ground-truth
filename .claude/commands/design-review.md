# Design Review Slash Command

## Command: /design-review

### Description
Triggers a comprehensive design review of recent changes or specified files/components.

### Usage
- `/design-review` - Review recent git changes
- `/design-review src/components/Button.tsx` - Review specific file
- `/design-review --all` - Review entire project

### What it does
1. Detects project framework and configuration
2. Analyzes component architecture and patterns
3. Checks accessibility compliance
4. Validates responsive design
5. Reviews design system adherence
6. Provides actionable feedback and suggestions

### Output
Structured report with:
- Overall design score
- Critical issues and warnings
- Accessibility concerns
- Responsive design feedback
- Code quality suggestions
- Best practice recommendations

