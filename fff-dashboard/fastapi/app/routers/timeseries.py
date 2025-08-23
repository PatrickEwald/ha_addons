from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, cast, Date
from datetime import datetime, date
from typing import List, Optional
import pytz
from ..db import get_db
from ..models import Gym, UsedCapacity, UtilizationForecast
from pydantic import BaseModel

router = APIRouter()

class TimeseriesResponse(BaseModel):
    mean: List[dict]
    forecast: List[dict]

@router.get("/gyms/{gym_uuid}/timeseries", response_model=TimeseriesResponse)
async def get_gym_timeseries(
    gym_uuid: str,
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    db: Session = Depends(get_db)
):
    """
    Get utilization timeseries data for a specific gym and date.
    Returns both actual values (mean) and forecast values.
    """
    try:
        # Parse the date parameter
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # Check if gym exists
    gym = db.query(Gym).filter(Gym.uuid == gym_uuid).first()
    if not gym:
        raise HTTPException(status_code=404, detail="Gym not found")
    
    # Get mean utilization data for the specified date
    mean_data = db.query(UsedCapacity).filter(
        UsedCapacity.gym_uuid == gym_uuid,
        cast(UsedCapacity.ts, Date) == target_date
    ).order_by(UsedCapacity.ts).all()
    
    # Get forecast data for the specified date
    forecast_data = db.query(UtilizationForecast).filter(
        and_(
            UtilizationForecast.gym_uuid == gym_uuid,
            UtilizationForecast.target_date == target_date
        )
    ).order_by(UtilizationForecast.ts).all()
    
    # Filter forecasts based on gym opening hours
    def is_within_opening_hours(timestamp, working_hours):
        """Check if timestamp is within gym opening hours"""
        if not working_hours:
            return True  # If no working hours defined, allow all times
            
        # Convert UTC timestamp to Berlin time for opening hours check
        berlin_tz = pytz.timezone('Europe/Berlin')
        berlin_dt = timestamp.astimezone(berlin_tz)
        day_name = berlin_dt.strftime("%a")
        
        if day_name not in working_hours:
            return False
            
        hours_str = working_hours[day_name]
        if not hours_str or '-' not in hours_str:
            return False
            
        try:
            open_str, close_str = hours_str.split('-')
            open_hour = int(open_str.split(':')[0])
            close_hour = int(close_str.split(':')[0])
            
            current_hour = berlin_dt.hour
            return open_hour <= current_hour <= close_hour
        except:
            return True  # If parsing fails, allow the forecast
    
    # Format the response
    mean_values = [
        {
            "timestamp": data.ts.isoformat(),
            "value": float(data.value),  # Changed from 'persons' to 'value' for dashboard compatibility
            "interval_sec": data.interval_sec
        }
        for data in mean_data
    ]
    
    # Filter forecast values by opening hours
    forecast_values = []
    for data in forecast_data:
        if is_within_opening_hours(data.ts, gym.working_hours):
            forecast_values.append({
                "timestamp": data.ts.isoformat(),
                "value": float(data.y_hat),  # Changed from 'persons' to 'value' for dashboard compatibility
                "interval_sec": data.interval_sec
            })
        # If outside opening hours, we could add a 0 value entry here if needed
        # but it's better to simply not include closed hours in the forecast
    
    return TimeseriesResponse(
        mean=mean_values,
        forecast=forecast_values
    )
