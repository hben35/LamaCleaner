from django.http import JsonResponse
from rest_framework.decorators import api_view
from pathlib import Path
from io import BytesIO
from PIL import Image
import requests
import numpy as np
import cv2
import base64

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

            img = url_to_image(input_image_url)
            if img is None:
                return JsonResponse({'status': 400, 'message': 'Failed to download or decode input image'}, safe=False)

            mask = url_to_image(mask_image_url, gray=True)
            if mask is None:
                return JsonResponse({'status': 400, 'message': 'Failed to download or decode mask image'}, safe=False)

            res = model(img, mask, get_config(HDStrategy.RESIZE))

            # Convertir l'image résultante en base64
            image_base64 = image_to_base64(res)

            response = {
                'status': 200,
                'message': "success",
                'image_base64': image_base64
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
    Télécharge une image depuis une URL et la convertit en une image PIL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        if gray:
            image = image.convert('L')
        return image
    except Exception as e:
        print(f"Error downloading or decoding image from {url}: {e}")
        return None

def image_to_base64(image):
    """
    Convertit une image PIL en base64.
    """
    try:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {e}")
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
