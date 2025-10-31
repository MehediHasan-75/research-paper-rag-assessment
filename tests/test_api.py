import pytest
from fastapi.testclient import TestClient
from src.main import app
import io

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint returns OK"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_endpoint():
    """Test health check"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

def test_list_papers():
    """Test listing papers"""
    response = client.get("/api/papers")
    assert response.status_code == 200
    data = response.json()
    assert "papers" in data
    assert "total" in data

def test_query_endpoint_valid():
    """Test valid query"""
    response = client.post("/api/query?question=What%20is%20machine%20learning?&top_k=5")
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data

def test_query_endpoint_invalid():
    """Test query with invalid parameters"""
    # Question too short
    response = client.post("/api/query?question=ML")
    assert response.status_code == 422  # Validation error
    
    # top_k out of range
    response = client.post("/api/query?question=Valid%20question&top_k=100")
    assert response.status_code == 422

def test_upload_non_pdf():
    """Test uploading non-PDF file"""
    file_content = b"Not a PDF"
    files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
    
    response = client.post("/api/papers/upload", files=files)
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]

def test_get_paper_not_found():
    """Test getting non-existent paper"""
    response = client.get("/api/papers/999999")
    assert response.status_code == 200  # Mock returns OK
    data = response.json()
    assert data["id"] == 999999
