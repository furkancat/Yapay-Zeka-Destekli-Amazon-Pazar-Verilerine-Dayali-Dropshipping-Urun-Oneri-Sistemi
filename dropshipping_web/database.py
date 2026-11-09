from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

VERITABANI_URL = "postgresql://postgres:@localhost/dropshipping2"

motor = create_engine(VERITABANI_URL)
OturumLocal = sessionmaker(autocommit=False, autoflush=False, bind=motor)
Taban = declarative_base()

def veritabani_al():
    db = OturumLocal()
    try:
        yield db
    finally:
        db.close()