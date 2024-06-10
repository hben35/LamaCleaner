from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
import datetime
from pathlib import Path
import cv2
import torch
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler, SDSampler

current_dir = Path(__file__).parent.absolute().resolve()
save_dir = current_dir / 'inpaintingresult'
save_dir.mkdir(exist_ok=True, parents=True)

@api_view(['POST'])
def lamaCleaner(request):
    if request.method == "POST":
        try:
            print("request.FILES content:", request.FILES)
            print("request.POST content:", request.POST)

            if 'input_image' not in request.FILES or 'mask_image' not in request.FILES or 'userid' not in request.POST:
                return JsonResponse({'status': 400, 'message': 'Missing required POST data'}, safe=False)

            input_image = request.FILES['input_image']
            mask_image = request.FILES['mask_image']
            userid = request.POST['userid']

            imagename = f"inpaint_{userid}.png"
            model = ModelManager(name="lama", device="cpu")

            input_image_path = current_dir / 'temp_input_image.jpg'
            mask_image_path = current_dir / 'temp_mask_image.jpg'

            with open(input_image_path, 'wb+') as temp_file:
                for chunk in input_image.chunks():
                    temp_file.write(chunk)

            with open(mask_image_path, 'wb+') as temp_file:
                for chunk in mask_image.chunks():
                    temp_file.write(chunk)

            print("Input image and mask image files saved.")

            img = cv2.imread(str(input_image_path))
            if img is None:
                print("Failed to read input image")
                return JsonResponse({'status': 400, 'message': 'Failed to read input image'}, safe=False)

            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            mask = cv2.imread(str(mask_image_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print("Failed to read mask image")
                return JsonResponse({'status': 400, 'message': 'Failed to read mask image'}, safe=False)

            print("Input image and mask image read successfully.")

            res = model(img, mask, get_config(HDStrategy.RESIZE))

            cv2.imwrite(
                str(save_dir / imagename),
                res,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100, int(cv2.IMWRITE_PNG_COMPRESSION), 0],
            )

            response = {
                'status': 200,
                'message': "success",
                'User': imagename
            }
            return JsonResponse(response, safe=False)

        except Exception as e:
            print("Exception occurred:", str(e))
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
