import asyncio
from io import BytesIO
from unittest import TestCase
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from PIL import Image
from starlette.datastructures import Headers

import backend.app.main as api


def upload(content: bytes, media_type: str = "image/png") -> UploadFile:
    return UploadFile(file=BytesIO(content), filename="image.png", headers=Headers({"content-type": media_type}))


class ApiValidationTests(TestCase):
    def test_rejects_invalid_image_content(self):
        with self.assertRaises(HTTPException) as error:
            api._decode_image(b"not an image")
        self.assertEqual(error.exception.status_code, 400)

    def test_rejects_upload_larger_than_limit(self):
        with patch.object(api, "MAX_UPLOAD_BYTES", 3):
            with self.assertRaises(HTTPException) as error:
                asyncio.run(api._read_limited_upload(upload(b"1234")))
        self.assertEqual(error.exception.status_code, 413)

    def test_predict_returns_the_public_result_only(self):
        buffer = BytesIO()
        Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
        model_result = {
            "label": "REAL",
            "decision_score": 0.9,
            "real_probability": 0.9,
            "fake_probability": 0.1,
            "decision_threshold": 0.4,
        }
        with patch("backend.app.main.predict_image", return_value=model_result):
            response = asyncio.run(api.predict(upload(buffer.getvalue())))
        self.assertEqual(response, model_result)
