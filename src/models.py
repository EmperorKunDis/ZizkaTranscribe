from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class AudioFile(Base):
    __tablename__ = "audio_files"
    
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    original_name = Column(String, nullable=False)
    status = Column(String, nullable=False)  # pending, processing, completed, error
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    model_type = Column(String, nullable=False)
    translation_language = Column(String, nullable=False)
    subtitle_format = Column(String, nullable=False)

# Database setup
engine = create_engine('sqlite:///audio_files.db')
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)
