# scripts/full_reset.py
import sys
import os
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.database import Base, engine, init_db
from src.services.qdrant_service import qdrant_service
from src.config import settings

def full_system_reset():
    print("🔥 Resetting everything...")
    
    # Drop database tables
    Base.metadata.drop_all(bind=engine)
    print("✅ Database tables dropped")
    
    # Delete uploaded PDFs
    if os.path.exists(settings.UPLOAD_DIR):
        shutil.rmtree(settings.UPLOAD_DIR)
        os.makedirs(settings.UPLOAD_DIR)
        print("✅ Uploaded files deleted")
    
    # Clear Qdrant
    try:
        qdrant_service.client.delete_collection("research_papers")
        print("✅ Qdrant vectors deleted")
    except:
        pass
    
    # Recreate everything
    init_db()
    qdrant_service._create_collection()
    print("✨ Reset complete!")

if __name__ == "__main__":
    confirm = input("Delete ALL data? (yes/no): ")
    if confirm.lower() == "yes":
        full_system_reset()
