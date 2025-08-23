from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import timeseries, gyms
from .db import engine
from .models import Base
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables with error handling
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.warning(f"Could not create database tables: {e}")
    logger.info("Tables may already exist or database connection not ready")

# Create FastAPI app
app = FastAPI(
    title="FFF-Docker API",
    description="API für Fitness First Auslastungsdaten und Vorhersagen",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(timeseries.router, prefix="/api/v1", tags=["timeseries"])
app.include_router(gyms.router, prefix="/api/v1", tags=["gyms"])

@app.get("/")
async def root():
    return {
        "message": "FFF-Docker API läuft!",
        "docs": "/docs",
        "endpoints": {
            "gyms": "/api/v1/gyms (Darmstadt only)",
            "gym_details": "/api/v1/gyms/{uuid} (Darmstadt only)",
            "timeseries": "/api/v1/gyms/{uuid}/timeseries?date=YYYY-MM-DD (returns absolute number of people)"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
