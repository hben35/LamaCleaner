from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view

import cv2
import base64
from io import BytesIO
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

            model = ModelManager(name="lama", device="cpu")
            
            img = cv2.imread(input_image_url)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_image_url, cv2.IMREAD_GRAYSCALE)

            res = model(img, mask, get_config(HDStrategy.RESIZE))

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
