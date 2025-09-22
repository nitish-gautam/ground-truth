# Universal Design Principles Template

## Design Philosophy
This template provides foundational design principles that can be adapted to any project. Copy this file to your project's `.claude/` directory and customize it for your specific needs.

## 1. Visual Hierarchy

### Typography
- **Primary Headings**: Clear distinction between h1-h6
- **Body Text**: Readable font sizes (minimum 16px)
- **Line Height**: 1.4-1.6 for optimal readability
- **Font Weights**: Consistent weight scale (300, 400, 600, 700)
- **Font Families**: Maximum 2-3 font families per project

### Spacing System
- **Base Unit**: 4px or 8px grid system
- **Consistent Margins**: Use spacing scale (4, 8, 16, 24, 32, 48, 64px)
- **Component Padding**: Internal spacing follows same scale
- **Section Gaps**: Vertical rhythm maintained throughout

## 2. Color System

### Brand Colors
- **Primary**: Main brand color for primary actions
- **Secondary**: Supporting brand color for variety
- **Accent**: Highlight color for emphasis

### Semantic Colors
- **Success**: Green tones for positive actions (#10b981, #059669)
- **Warning**: Amber/Orange for cautions (#f59e0b, #d97706)
- **Error**: Red tones for errors (#ef4444, #dc2626)
- **Info**: Blue tones for information (#0ea5e9, #0284c7)

### Neutral Palette
- **Text Primary**: High contrast for main text
- **Text Secondary**: Medium contrast for supporting text
- **Text Disabled**: Low contrast for disabled states
- **Backgrounds**: Multiple levels for depth and hierarchy
- **Borders**: Subtle lines for separation

### Accessibility Requirements
- **Contrast Ratios**: Minimum 4.5:1 for normal text, 3:1 for large text
- **Color Independence**: Information conveyed without color alone
- **Dark Mode**: Complete color system for dark theme

## 3. Component Design

### Interactive Elements
- **Buttons**: Clear hierarchy (primary, secondary, tertiary)
- **Links**: Distinct from regular text, consistent styling
- **Form Controls**: Clear labels, validation states, focus indicators
- **Navigation**: Active states, clear organization

### State Management
- **Default**: Normal state styling
- **Hover**: Subtle feedback on interactive elements
- **Focus**: Clear focus indicators for keyboard navigation
- **Active**: Pressed/clicked state feedback
- **Disabled**: Visually distinct disabled appearance
- **Loading**: Clear loading indicators

### Component Architecture
- **Reusability**: Components should be modular and reusable
- **Consistency**: Similar components should behave similarly
- **Flexibility**: Support variants without breaking patterns
- **Composition**: Prefer composition over complex single components

## 4. Layout Principles

### Grid System
- **Responsive Grid**: 12-column or flexible grid system
- **Breakpoints**: Mobile-first responsive design
  - Mobile: 0-768px
  - Tablet: 768-1024px
  - Desktop: 1024px+
- **Container Widths**: Maximum widths for content areas
- **Gutter Sizes**: Consistent spacing between grid elements

### Content Organization
- **Scannable**: Use headings, lists, and white space effectively
- **Logical Flow**: Information hierarchy follows user needs
- **Progressive Disclosure**: Complex information revealed progressively
- **Visual Grouping**: Related content visually grouped together

## 5. Accessibility Standards

### WCAG AA Compliance
- **Keyboard Navigation**: All interactive elements keyboard accessible
- **Screen Readers**: Proper ARIA labels and semantic HTML
- **Focus Management**: Logical tab order and focus trapping
- **Alternative Text**: Descriptive alt text for images
- **Captions**: Video/audio content properly captioned

### Inclusive Design
- **Motor Accessibility**: Large enough touch targets (44px minimum)
- **Cognitive Accessibility**: Clear language and consistent patterns
- **Visual Accessibility**: Support for zoom up to 200%
- **Reduced Motion**: Respect prefers-reduced-motion settings

## 6. Performance Standards

### Loading Performance
- **Critical CSS**: Above-fold styles loaded first
- **Font Loading**: Web font optimization and fallbacks
- **Image Optimization**: Proper formats and lazy loading
- **Code Splitting**: JavaScript bundles optimized for loading

### Runtime Performance
- **Smooth Animations**: 60fps animations, proper timing
- **Responsive Interactions**: <100ms response to user input
- **Memory Management**: Efficient component lifecycle management
- **Bundle Size**: Reasonable JavaScript bundle sizes

## 7. Content Guidelines

### Microcopy
- **Clear Labels**: Descriptive form labels and button text
- **Error Messages**: Helpful, actionable error messages
- **Empty States**: Friendly messages for empty content
- **Loading States**: Clear progress indicators with context

### Tone and Voice
- **Consistency**: Consistent voice throughout application
- **Clarity**: Clear, concise language
- **Helpfulness**: Supportive and guiding tone
- **Accessibility**: Plain language principles

## 8. Animation and Motion

### Purpose-Driven Animation
- **Feedback**: Animations provide user feedback
- **Transition**: Smooth state transitions
- **Attention**: Direct attention without being distracting
- **Delight**: Subtle moments of delight and personality

### Technical Standards
- **Performance**: Hardware-accelerated animations (transform, opacity)
- **Duration**: Appropriate timing (200-500ms for most UI animations)
- **Easing**: Natural easing curves (ease-out for exits, ease-in for entrances)
- **Respect Preferences**: Honor prefers-reduced-motion

## 9. Responsive Design

### Mobile-First Approach
- **Progressive Enhancement**: Start with mobile, enhance for larger screens
- **Touch Targets**: Minimum 44px tap targets
- **Content Priority**: Most important content accessible on mobile
- **Performance**: Optimized for mobile networks

### Breakpoint Strategy
- **Fluid Design**: Layout adapts smoothly between breakpoints
- **Content Reorganization**: Layout changes serve content effectively
- **Image Responsiveness**: Images scale appropriately
- **Navigation Adaptation**: Navigation patterns work on all devices

## 10. Error Handling and Edge Cases

### Error States
- **Graceful Degradation**: System works when components fail
- **Clear Messages**: Errors explained in user-friendly language
- **Recovery Options**: Users can recover from errors
- **Consistent Styling**: Error states follow design system

### Edge Cases
- **Empty States**: Designed states for no content
- **Loading States**: Progressive loading indicators
- **Offline States**: Graceful offline experience
- **Long Content**: Handling of very long text or lists

## Customization Instructions

### Project Adaptation
1. **Copy this template** to your project's `.claude/design-principles.md`
2. **Customize colors** to match your brand palette
3. **Adjust spacing** to fit your design system
4. **Add specific components** relevant to your project
5. **Include examples** from your actual codebase
6. **Define project-specific patterns** and conventions

### Framework-Specific Considerations
- **React**: Component composition, hooks patterns, prop design
- **Vue**: Template patterns, reactivity considerations, slot usage
- **Angular**: Component lifecycle, dependency injection patterns
- **CSS Framework**: Specific utility patterns (Tailwind, styled-components)

### Industry Adaptations
- **SaaS**: Dashboard patterns, data visualization, user management
- **E-commerce**: Product displays, checkout flows, responsive images
- **Content**: Reading experience, typography, content hierarchy
- **Mobile Apps**: Touch interactions, native-like patterns