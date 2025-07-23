from fastapi import FastAPI
from app.api.v1.api import api_router

# Create the main FastAPI application instance
app = FastAPI(
    title="Options Scalping Platform API",
    description="Provides real-time signal analysis and trading logic.",
    version="1.0.0"
)

# Include the main API router
# All routes from api_router will be prefixed with /api/v1
app.include_router(api_router, prefix="/api/v1")

@app.get("/", summary="Health Check", tags=["Health"])
def read_root():
    """
    Root endpoint for basic health checks.
    """
    return {"status": "ok", "message": "Welcome to the Scalping Platform API"}

# In a real application, you would also configure middleware here for:
# - CORS (Cross-Origin Resource Sharing) to allow the frontend to connect
# - Authentication
# - Logging
# - etc.

