from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view

import cv2
import requests
import numpy as np
import base64

from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler


@api_view(['POST'])
def lamaCleaner(request):
    if request.method == "POST":
        try:
            input_image_url = request.POST.get('input_image')
            mask_image_url = request.POST.get('mask_image')
            userid = request.POST.get('userid')

            if not input_image_url or not mask_image_url or not userid:
                return JsonResponse({'status': 400, 'message': 'Missing required fields'}, safe=False)

            model = ModelManager(name="lama", device="cpu", dtype="float32")
            

            img = url_to_image(input_image_url)
            if img is None:
                return JsonResponse({'status': 400, 'message': 'Failed to download or read input image'}, safe=False)

            mask = url_to_image(mask_image_url, gray=True)
            if mask is None:
                return JsonResponse({'status': 400, 'message': 'Failed to download or read mask image'}, safe=False)
            
            print(f"Original image size: {img.shape}")
            print(f"Mask image size: {mask.shape}")
            
            # Redimensionnement manuel avant traitement
            original_size = img.shape[:2]
            target_size = (512, 512)  # Taille à adapter en fonction de vos tests
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

            # Effectuer l'inpainting
            res_resized = model(img_resized, mask_resized, get_config(HDStrategy.CROP))

            # Redimensionner l'image résultante à la taille originale
            res = cv2.resize(res_resized, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)

            # Convertir l'image résultante en base64
            _, buffer = cv2.imencode('.png', res)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            response = {
                'status': 200,
                'message': "success",
                'image_base64': "data:image/png;base64," + image_base64
            }

            return JsonResponse(response, safe=False)

        except Exception as e:
            response = {
                'status': 500,
                'message': str(e)
            }
            return JsonResponse(response, safe=False)



def url_to_image(url, gray=False):
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
        hd_strategy_crop_trigger_size=512,  # Essayer une valeur plus grande
        hd_strategy_resize_limit=2048,  # Augmenter la limite de redimensionnement
    )
    data.update(**kwargs)
    return Config(**data)
