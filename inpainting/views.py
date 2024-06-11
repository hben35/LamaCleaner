from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
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
            input_image_url = request.POST.get('input_image')
            mask_image_url = request.POST.get('mask_image')
            userid = request.POST.get('userid')

            if not input_image_url or not mask_image_url or not userid:
                return JsonResponse({'status': 400, 'message': 'Missing required fields'}, safe=False)

            model = ModelManager(name="lama", device="cpu")

            # Download and read images
            img = url_to_image(input_image_url)
            if img is None:
                return JsonResponse({'status': 400, 'message': 'Failed to download or read input image'}, safe=False)

            mask = url_to_image(mask_image_url, gray=True)
            if mask is None:
                return JsonResponse({'status': 400, 'message': 'Failed to download or read mask image'}, safe=False)
            
            try:
                # Calculate Scaling Factor (if needed)
                max_size = getattr(settings, 'LAMA_CLEANER_MAX_SIZE', 1024)
                scale_factor = min(1.0, max_size / max(img.shape[0], img.shape[1]))

                if scale_factor < 1.0:
                    img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
                    mask = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor)

                # Adjusted Configuration for Large Images
                config = get_config(
                    HDStrategy.RESIZE, 
                    hd_strategy_resize_limit=max_size  # Set resize limit if needed
                )
                res = model(img, mask, config)

                # Optimized Image Conversion
                with BytesIO() as image_buffer:
                    Image.fromarray(res).save(image_buffer, format='PNG')
                    image_base64 = base64.b64encode(image_buffer.getvalue()).decode()

                response = {
                    'status': 200,
                    'message': "success",
                    'image_base64': "data:image/png;base64," + image_base64
                }
            except MemoryError:
                response = {
                    'status': 500,
                    'message': "Image too large to process. Please try a smaller image or reduce image resolution."
                }

            return JsonResponse(response, safe=False)

        except Exception as e:
            response = {
                'status': 500,
                'message': str(e)
            }
            return JsonResponse(response, safe=False)


def url_to_image(url, gray=False):
    """
    Télécharge une image depuis une URL et la convertit en un format OpenCV.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        if image is not None and not gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None


def get_config(strategy, **kwargs):
    data = dict(
        ldm_steps=10,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=strategy,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=512,
        hd_strategy_resize_limit=1024,
    )
    data.update(**kwargs)
    return Config(**data)
