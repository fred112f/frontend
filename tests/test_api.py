"""Unit tests for the emotion classification API."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from http import HTTPStatus

from exam_project.api import app


@pytest.fixture
def client():
    """Create a test client for the API."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_image():
    """Find a sample image for testing."""
    data_dir = Path("data/raw/datasets/msambare/fer2013/versions/1/train")
    if data_dir.exists():
        for emotion_dir in sorted(data_dir.iterdir()):
            if emotion_dir.is_dir():
                for img in emotion_dir.glob('*.jpg'):
                    return img
    return None


class TestHeaderValidation:
    """Test API header validation."""

    def test_missing_authorization_header(self, client, sample_image):
        """Test that request without authorization header is rejected."""
        if sample_image is None:
            pytest.skip("No sample image found")

        with open(sample_image, 'rb') as f:
            response = client.post(
                "/predict/",
                files={"file": f},
                headers={"accept": "application/json"}
            )

        # Should get 401 or 400 error
        assert response.status_code in [
            HTTPStatus.UNAUTHORIZED,
            HTTPStatus.BAD_REQUEST
        ]

    def test_invalid_authorization_header(self, client, sample_image):
        """Test that request with invalid authorization is rejected."""
        if sample_image is None:
            pytest.skip("No sample image found")

        with open(sample_image, 'rb') as f:
            response = client.post(
                "/predict/",
                files={"file": f},
                headers={
                    "authorization": "invalid-key",
                    "accept": "application/json"
                }
            )

        # Should get 401 error for invalid auth
        assert response.status_code == HTTPStatus.UNAUTHORIZED

    def test_missing_accept_header(self, client, sample_image):
        """Test that request without accept header is rejected."""
        if sample_image is None:
            pytest.skip("No sample image found")

        with open(sample_image, 'rb') as f:
            response = client.post(
                "/predict/",
                files={"file": f},
                headers={"authorization": "dtu"}
            )

        # Should get 400 error for missing accept
        assert response.status_code == HTTPStatus.BAD_REQUEST

    def test_invalid_accept_header(self, client, sample_image):
        """Test that request with invalid accept header is rejected."""
        if sample_image is None:
            pytest.skip("No sample image found")

        with open(sample_image, 'rb') as f:
            response = client.post(
                "/predict/",
                files={"file": f},
                headers={
                    "authorization": "dtu",
                    "accept": "text/html"
                }
            )

        assert response.status_code == HTTPStatus.BAD_REQUEST


class TestSuccessfulPrediction:
    """Test successful API predictions."""

    def test_predict_success_returns_valid_response(self, client, sample_image):
        """Test successful prediction with correct headers."""
        if sample_image is None:
            pytest.skip("No sample image found")

        with open(sample_image, 'rb') as f:
            response = client.post(
                "/predict/",
                files={"file": f},
                headers={
                    "authorization": "dtu",
                    "accept": "application/json"
                }
            )

        assert response.status_code == HTTPStatus.OK
        result = response.json()

        # Check all required fields are present
        required_fields = {
            "emotion", "confidence", "probabilities", "status-code", "message"
        }
        assert required_fields.issubset(result.keys())

        # Check response metadata
        assert result["status-code"] == 200
        assert result["message"] == "OK"

    def test_predict_response_is_json(self, client, sample_image):
        """Test that response content-type is JSON."""
        if sample_image is None:
            pytest.skip("No sample image found")

        with open(sample_image, 'rb') as f:
            response = client.post(
                "/predict/",
                files={"file": f},
                headers={
                    "authorization": "dtu",
                    "accept": "application/json"
                }
            )

        assert response.status_code == HTTPStatus.OK
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type


class TestFileHandling:
    """Test API file handling."""

    def test_predict_no_file(self, client):
        """Test that request without file is rejected."""
        response = client.post(
            "/predict/",
            headers={
                "authorization": "dtu",
                "accept": "application/json"
            }
        )

        # Should return error when no file is provided
        assert response.status_code in [
            HTTPStatus.BAD_REQUEST,
            HTTPStatus.UNPROCESSABLE_ENTITY
        ]

    def test_predict_with_valid_file(self, client, sample_image):
        """Test that valid file is accepted."""
        if sample_image is None:
            pytest.skip("No sample image found")

        with open(sample_image, 'rb') as f:
            response = client.post(
                "/predict/",
                files={"file": f},
                headers={
                    "authorization": "dtu",
                    "accept": "application/json"
                }
            )

        assert response.status_code == HTTPStatus.OK


class TestConsistency:
    """Test API consistency and reproducibility."""

    def test_same_image_produces_same_prediction(self, client, sample_image):
        """Test that the same image produces the same prediction."""
        if sample_image is None:
            pytest.skip("No sample image found")

        # First prediction
        with open(sample_image, 'rb') as f:
            response1 = client.post(
                "/predict/",
                files={"file": f},
                headers={
                    "authorization": "dtu",
                    "accept": "application/json"
                }
            )
        result1 = response1.json()

        # Second prediction with same image
        with open(sample_image, 'rb') as f:
            response2 = client.post(
                "/predict/",
                files={"file": f},
                headers={
                    "authorization": "dtu",
                    "accept": "application/json"
                }
            )
        result2 = response2.json()

        # Both should produce identical results
        assert result1["emotion"] == result2["emotion"]
        assert result1["confidence"] == result2["confidence"]
        assert result1["probabilities"] == result2["probabilities"]
