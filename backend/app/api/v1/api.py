from fastapi import APIRouter
from app.api.v1.endpoints.signals import router as signals_router

# Create the main router for the v1 API
api_router = APIRouter()

# --- FIX: Use the directly imported router object ---
api_router.include_router(
    signals_router, 
    prefix="/signals",
    tags=["Signals"] # Add a tag for better organization in the docs
)

# You can add other routers here in the future
# For example:
# from app.api.v1.endpoints import positions
# api_router.include_router(positions.router, prefix="/positions", tags=["Positions"])

