"""Unit tests for the emotion classification API."""

import os
import pytest
import requests
from pathlib import Path
from http import HTTPStatus

# Use cloud API or local API
API_URL = os.getenv(
    "API_URL",
    "https://emotion-classifier-597500488480.europe-west1.run.app"
)


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

    def test_missing_authorization_header(self, sample_image):
        """Test that request without authorization header is rejected."""
        if sample_image is None:
            pytest.skip("No sample image found")

        with open(sample_image, 'rb') as f:
            response = requests.post(
                f"{API_URL}/predict/",
                files={"file": f},
                headers={"accept": "application/json"},
                timeout=30
            )

        # Should get 401 or 400 error
        assert response.status_code in [
            HTTPStatus.UNAUTHORIZED,
            HTTPStatus.BAD_REQUEST
        ]

    def test_invalid_authorization_header(self, sample_image):
        """Test that request with invalid authorization is rejected."""
        if sample_image is None:
            pytest.skip("No sample image found")

        with open(sample_image, 'rb') as f:
            response = requests.post(
                f"{API_URL}/predict/",
                files={"file": f},
                headers={
                    "authorization": "invalid-key",
                    "accept": "application/json"
                },
                timeout=30
            )

        # Should get 401 error for invalid auth
        assert response.status_code == HTTPStatus.UNAUTHORIZED

    def test_missing_accept_header(self, sample_image):
        """Test that request without accept header is rejected."""
        if sample_image is None:
            pytest.skip("No sample image found")

        with open(sample_image, 'rb') as f:
            response = requests.post(
                f"{API_URL}/predict/",
                files={"file": f},
                headers={"authorization": "dtu"},
                timeout=30
            )

        # Should get 400 error for missing accept
        assert response.status_code == HTTPStatus.BAD_REQUEST

    def test_invalid_accept_header(self, sample_image):
        """Test that request with invalid accept header is rejected."""
        if sample_image is None:
            pytest.skip("No sample image found")

        with open(sample_image, 'rb') as f:
            response = requests.post(
                f"{API_URL}/predict/",
                files={"file": f},
                headers={
                    "authorization": "dtu",
                    "accept": "text/html"
                },
                timeout=30
            )

        assert response.status_code == HTTPStatus.BAD_REQUEST


class TestSuccessfulPrediction:
    """Test successful API predictions."""

    def test_predict_success_returns_valid_response(self, sample_image):
        """Test successful prediction with correct headers."""
        if sample_image is None:
            pytest.skip("No sample image found")

        with open(sample_image, 'rb') as f:
            response = requests.post(
                f"{API_URL}/predict/",
                files={"file": f},
                headers={
                    "authorization": "dtu",
                    "accept": "application/json"
                },
                timeout=30
            )

        assert response.status_code == HTTPStatus.OK
        result = response.json()

        # Check all required fields are present
        required_fields = {
            "emotion", "confidence", "probabilities", "message"
        }
        assert required_fields.issubset(result.keys())

        # Check emotion is valid
        valid_emotions = {"angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"}
        assert result["emotion"] in valid_emotions

        # Check confidence is between 0 and 1
        assert 0 <= result["confidence"] <= 1

    def test_predict_response_is_json(self, sample_image):
        """Test that response content-type is JSON."""
        if sample_image is None:
            pytest.skip("No sample image found")

        with open(sample_image, 'rb') as f:
            response = requests.post(
                f"{API_URL}/predict/",
                files={"file": f},
                headers={
                    "authorization": "dtu",
                    "accept": "application/json"
                },
                timeout=30
            )

        assert response.status_code == HTTPStatus.OK
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type


class TestFileHandling:
    """Test API file handling."""

    def test_predict_no_file(self):
        """Test that request without file is rejected."""
        response = requests.post(
            f"{API_URL}/predict/",
            headers={
                "authorization": "dtu",
                "accept": "application/json"
            },
            timeout=30
        )

        # Should return error when no file is provided
        assert response.status_code in [
            HTTPStatus.BAD_REQUEST,
            HTTPStatus.UNPROCESSABLE_ENTITY
        ]

    def test_predict_with_valid_file(self, sample_image):
        """Test that valid file is accepted."""
        if sample_image is None:
            pytest.skip("No sample image found")

        with open(sample_image, 'rb') as f:
            response = requests.post(
                f"{API_URL}/predict/",
                files={"file": f},
                headers={
                    "authorization": "dtu",
                    "accept": "application/json"
                },
                timeout=30
            )

        assert response.status_code == HTTPStatus.OK


class TestConsistency:
    """Test API consistency and reproducibility."""

    def test_same_image_produces_same_prediction(self, sample_image):
        """Test that the same image produces the same prediction."""
        if sample_image is None:
            pytest.skip("No sample image found")

        # First prediction
        with open(sample_image, 'rb') as f:
            response1 = requests.post(
                f"{API_URL}/predict/",
                files={"file": f},
                headers={
                    "authorization": "dtu",
                    "accept": "application/json"
                },
                timeout=30
            )
        result1 = response1.json()

        # Second prediction with same image
        with open(sample_image, 'rb') as f:
            response2 = requests.post(
                f"{API_URL}/predict/",
                files={"file": f},
                headers={
                    "authorization": "dtu",
                    "accept": "application/json"
                },
                timeout=30
            )
        result2 = response2.json()

        # Both should produce identical results
        assert result1["emotion"] == result2["emotion"]
        assert result1["confidence"] == result2["confidence"]
        assert result1["probabilities"] == result2["probabilities"]
