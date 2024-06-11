from django.http import JsonResponse
from rest_framework.decorators import api_view
from pathlib import Path
from io import BytesIO
import requests
import numpy as np
import base64
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

            img = url_to_image(input_image_url)
            if img is None:
                return JsonResponse({'status': 400, 'message': 'Failed to download or decode input image'}, safe=False)

            mask = url_to_image(mask_image_url)
            if mask is None:
                return JsonResponse({'status': 400, 'message': 'Failed to download or decode mask image'}, safe=False)

            # Assurer que l'image et le masque ont les mêmes dimensions
            img, mask = resize_to_same_dimension(img, mask)

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

def url_to_image(url):
    """
    Télécharge une image depuis une URL et la convertit en un format Pillow.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def resize_to_same_dimension(img, mask):
    """
    Redimensionne l'image et le masque pour avoir les mêmes dimensions.
    """
    min_height = min(img.height, mask.height)
    min_width = min(img.width, mask.width)
    img = img.resize((min_width, min_height))
    mask = mask.resize((min_width, min_height))
    return img, mask

def image_to_base64(image):
    """
    Convertit une image Pillow en format base64.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

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
