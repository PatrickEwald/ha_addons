from sqlalchemy import Column, Integer, String, DateTime, Numeric, Date, Text, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .db import Base
import uuid

class Gym(Base):
    __tablename__ = "gyms"
    
    uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    timezone = Column(String(50), nullable=False, default='UTC')
    working_hours = Column(JSONB, nullable=False, default={})
    total_capacity = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class UsedCapacity(Base):
    __tablename__ = "used_capacity"
    
    id = Column(Integer, primary_key=True)
    gym_uuid = Column(UUID(as_uuid=True), ForeignKey('gyms.uuid', ondelete='CASCADE'), nullable=False)
    ts = Column(DateTime(timezone=True), nullable=False)
    interval_sec = Column(Integer, nullable=False, default=300)
    value = Column(Numeric(5,2), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    gym = relationship("Gym", backref="used_capacities")
    
    # Unique constraint
    __table_args__ = (UniqueConstraint('gym_uuid', 'ts', 'interval_sec', name='uq_used_capacity_gym_ts_interval'),)

class UtilizationForecast(Base):
    __tablename__ = "utilization_forecast"
    
    id = Column(Integer, primary_key=True)
    gym_uuid = Column(UUID(as_uuid=True), ForeignKey('gyms.uuid', ondelete='CASCADE'), nullable=False)
    ts = Column(DateTime(timezone=True), nullable=False)
    target_date = Column(Date, nullable=False)
    interval_sec = Column(Integer, nullable=False, default=3600)
    y_hat = Column(Numeric(5,2), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    gym = relationship("Gym", backref="utilization_forecasts")
    
    # Unique constraint
    __table_args__ = (UniqueConstraint('gym_uuid', 'ts', 'target_date', 'interval_sec', name='uq_utilization_forecast_gym_ts_target_interval'),)
