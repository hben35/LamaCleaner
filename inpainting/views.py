from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view

from pathlib import Path
import cv2
import requests
import numpy as np
import base64
from io import BytesIO
from PIL import Image

from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler, SDSampler


@api_view(['POST'])
def lamaCleaner(request):
    if request.method == "POST":
        try:
            input_image_base64 = request.POST.get('input_image_base64')
            mask_image_base64 = request.POST.get('mask_image_base64')
            userid = request.POST.get('userid')

            if not input_image_base64 or not mask_image_base64 or not userid:
                return JsonResponse({'status': 400, 'message': 'Missing required fields'}, safe=False)

            model = ModelManager(name="lama", device="cpu")

            # Decode base64-encoded images
            input_image = decode_base64_to_cv2(input_image_base64)
            mask_image = decode_base64_to_cv2(mask_image_base64, gray=True)

            if input_image is None or mask_image is None:
                return JsonResponse({'status': 400, 'message': 'Failed to decode input images'}, safe=False)

            # Perform inpainting
            result_image = model(input_image, mask_image, get_config(HDStrategy.RESIZE))

            # Convert result image to base64
            _, buffer = cv2.imencode('.png', result_image)
            result_image_base64 = base64.b64encode(buffer).decode('utf-8')

            response = {
                'status': 200,
                'message': "success",
                'image_base64': "data:image/png;base64," + result_image_base64
            }

            return JsonResponse(response, safe=False)

        except Exception as e:
            response = {
                'status': 500,
                'message': str(e)
            }
            return JsonResponse(response, safe=False)


def decode_base64_to_cv2(base64_string, gray=False):
    """
    Decode base64-encoded image to OpenCV format.
    """
    try:
        decoded_data = base64.b64decode(base64_string)
        image_array = np.frombuffer(decoded_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        if image is not None and not gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None


def get_config(strategy, **kwargs):
    data = dict(
        ldm_steps=1,
        ldm_sampler=LDMSampler.plms,
        hd_strategy=strategy,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=200,
        hd_strategy_resize_limit=200,
    )
    data.update(**kwargs)
    return Config(**data)
