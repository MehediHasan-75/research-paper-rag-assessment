from sqlalchemy import text
from src.models.database import engine

def fix_schema():
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE chunks ALTER COLUMN section TYPE VARCHAR(500)"))
        conn.commit()
        print("✅ Schema updated successfully")

if __name__ == "__main__":
    fix_schema()
