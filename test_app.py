"""
Simple FastAPI test application to verify setup
"""

from fastapi import FastAPI

app = FastAPI(
    title="Underground Utility Detection Platform - Test",
    description="Test API to verify environment setup",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Underground Utility Detection Platform API - Test Mode",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Underground Utility Detection Platform - Test",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)