from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from ..db import get_db
from ..models import Gym
from pydantic import BaseModel

router = APIRouter()

class GymResponse(BaseModel):
    uuid: str
    name: str
    timezone: str
    working_hours: dict
    total_capacity: int
    created_at: str
    updated_at: str

@router.get("/gyms", response_model=List[GymResponse])
async def get_all_gyms(db: Session = Depends(get_db)):
    """
    Get all available gyms/fitness studios in Darmstadt
    """
    try:
        # Filter for Darmstadt studios only
        gyms = db.query(Gym).filter(
            Gym.name.ilike('%darmstadt%') | 
            Gym.name.ilike('%darms%')
        ).all()
        
        if not gyms:
            return []
        
        return [
            GymResponse(
                uuid=str(gym.uuid),
                name=gym.name,
                timezone=gym.timezone,
                working_hours=gym.working_hours,
                total_capacity=gym.total_capacity,
                created_at=gym.created_at.isoformat() if gym.created_at else None,
                updated_at=gym.updated_at.isoformat() if gym.updated_at else None
            )
            for gym in gyms
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/gyms/{gym_uuid}", response_model=GymResponse)
async def get_gym_by_uuid(gym_uuid: str, db: Session = Depends(get_db)):
    """
    Get a specific gym by UUID (Darmstadt studios only)
    """
    try:
        gym = db.query(Gym).filter(Gym.uuid == gym_uuid).first()
        
        if not gym:
            raise HTTPException(status_code=404, detail="Gym not found")
        
        return GymResponse(
            uuid=str(gym.uuid),
            name=gym.name,
            timezone=gym.timezone,
            working_hours=gym.working_hours,
            total_capacity=gym.total_capacity,
            created_at=gym.created_at.isoformat() if gym.created_at else None,
            updated_at=gym.updated_at.isoformat() if gym.updated_at else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
