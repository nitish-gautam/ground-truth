# ============================================
# Multi-Stage Dockerfile for React Frontend
# ============================================
# Stage 1: Development with hot-reload
# Stage 2: Production build

# ============================================
# BASE STAGE - Node.js
# ============================================
FROM node:20-alpine as base

# Set working directory
WORKDIR /app

# Install curl for healthcheck
RUN apk add --no-cache curl

# ============================================
# DEPENDENCIES STAGE
# ============================================
FROM base as dependencies

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci

# ============================================
# DEVELOPMENT STAGE
# ============================================
FROM dependencies as development

# Copy all source files
COPY . .

# Expose Vite dev server port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Start Vite dev server with hot-reload
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]

# ============================================
# BUILD STAGE - Production build
# ============================================
FROM dependencies as build

# Copy source files
COPY . .

# Build the application
RUN npm run build

# ============================================
# PRODUCTION STAGE - Serve with nginx
# ============================================
FROM nginx:alpine as production

# Copy built assets from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
