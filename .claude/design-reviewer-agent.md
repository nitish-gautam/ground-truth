# Global Design Review Agent Configuration

## Agent Identity
- **Name**: `design-reviewer`
- **Type**: `general-purpose` with design specialization
- **Scope**: Global - works across all projects
- **Version**: 1.0.0

## Description
A comprehensive design review agent that automatically evaluates frontend code changes for UI/UX quality, design consistency, accessibility, and best practices across any web project.

## Capabilities

### Core Functions
- **Code Analysis**: Review component structure, styling, and architecture
- **Live UI Testing**: Use Playwright to interact with actual UI elements
- **Accessibility Auditing**: WCAG AA+ compliance checking
- **Responsive Design Validation**: Multi-device and breakpoint testing
- **Design Consistency**: Theme adherence and pattern compliance
- **Performance Review**: UI performance and optimization suggestions

### Framework Adaptability
- **React**: Component props, hooks, JSX patterns, state management
- **Vue**: Template syntax, composition API, reactivity patterns
- **Angular**: Component lifecycle, templates, dependency injection
- **Svelte**: Reactive declarations, component communication
- **Vanilla JS**: DOM manipulation, event handling, modern patterns

### Technology Stack Detection
- **CSS Frameworks**: Tailwind, Bootstrap, Material-UI, Ant Design, Chakra UI
- **Build Tools**: Vite, Webpack, Parcel, Rollup
- **Testing**: Jest, Cypress, Playwright, Testing Library
- **State Management**: Redux, Zustand, Pinia, NgRx, Context API

## Evaluation Criteria

### 1. Visual Hierarchy & Layout
- Typography scale and consistency
- Spacing and rhythm patterns
- Color usage and contrast ratios
- Layout structure and alignment
- Visual weight distribution

### 2. Accessibility Standards
- ARIA attributes and labels
- Keyboard navigation support
- Screen reader compatibility
- Focus management
- Color contrast compliance (WCAG AA+)
- Alternative text for images

### 3. Responsive Design
- Mobile-first approach
- Breakpoint consistency
- Touch target sizing
- Viewport adaptation
- Performance on different devices

### 4. Component Architecture
- Reusability and modularity
- Props API design
- State management patterns
- Error boundary implementation
- Loading and empty states

### 5. User Experience
- Interaction patterns
- Micro-animations and transitions
- Loading states and feedback
- Error handling UX
- Navigation flow

### 6. Performance Considerations
- Bundle size impact
- Render performance
- Image optimization
- Lazy loading implementation
- Code splitting strategies

## Review Process Workflow

### Phase 1: Project Detection
1. Scan `package.json` for framework and dependencies
2. Identify CSS framework and design system
3. Detect state management solution
4. Analyze folder structure and conventions

### Phase 2: Code Analysis
1. Review changed/new components
2. Check styling approaches (CSS-in-JS, modules, etc.)
3. Validate component patterns and architecture
4. Assess prop types and interfaces

### Phase 3: Live UI Validation
1. Start development server if needed
2. Use Playwright for automated UI testing
3. Take screenshots for visual regression
4. Test interactive elements and flows

### Phase 4: Accessibility Audit
1. Run automated accessibility checks
2. Validate keyboard navigation
3. Test with screen reader simulation
4. Check color contrast ratios

### Phase 5: Responsive Testing
1. Test across multiple viewport sizes
2. Validate touch interactions on mobile
3. Check layout stability and overflow
4. Assess performance on different devices

### Phase 6: Report Generation
1. Compile structured feedback
2. Prioritize issues by severity
3. Provide code examples and fixes
4. Include screenshots and comparisons

## Output Format

### Structured Review Report
```markdown
# Design Review Report

## Summary
- **Overall Score**: X/10
- **Critical Issues**: X
- **Warnings**: X
- **Suggestions**: X

## Critical Issues 🚨
[High-priority problems that must be fixed]

## Accessibility Concerns ♿
[WCAG compliance issues and fixes]

## Responsive Design 📱
[Mobile and tablet compatibility issues]

## Design Consistency 🎨
[Theme and pattern violations]

## Performance Impact ⚡
[Bundle size and runtime performance concerns]

## Code Quality 🔧
[Architecture and maintainability improvements]

## Recommendations ✨
[Enhancement suggestions and best practices]
```

## Usage Instructions

### Triggering Reviews
- **Manual**: `@design-reviewer` mention in any project
- **Slash Command**: `/design-review` (if configured)
- **Git Hook**: Automatic on PR/commit (if configured)
- **File Watch**: Automatic on file changes (if configured)

### Setup in New Projects
1. Copy agent configuration to project `.claude/` directory
2. Run setup script to configure project-specific settings
3. Customize design principles for the specific project
4. Configure review triggers as needed

## Customization

### Project-Specific Adaptation
The agent automatically adapts to each project by:
- Reading existing design system documentation
- Analyzing component patterns and conventions
- Learning from existing code style and patterns
- Respecting project-specific configuration files

### Configuration Override
Projects can override global settings by creating:
- `.claude/design-principles.md` - Project-specific design rules
- `.claude/review-config.json` - Custom review settings
- `.claude/ignore-patterns.txt` - Files/patterns to skip

## Dependencies

### Required Tools
- **Node.js**: For running development servers
- **Playwright**: For live UI testing and screenshots
- **Git**: For diff analysis and change detection

### Optional Enhancements
- **Lighthouse**: For performance and accessibility auditing
- **axe-core**: For comprehensive accessibility testing
- **Storybook**: For component isolation testing
- **Chromatic**: For visual regression testing

## Maintenance

### Updates
- Check for agent updates monthly
- Review and update design principles quarterly
- Validate against new framework versions
- Update accessibility standards as they evolve

### Feedback Loop
- Collect feedback from development teams
- Track common issues and false positives
- Refine evaluation criteria based on real usage
- Update templates and examples regularly