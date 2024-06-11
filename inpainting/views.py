from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view

import cv2
import requests
import numpy as np
import base64

@api_view(['POST'])
def lamaCleaner(request):
    if request.method == "POST":
        try:
            input_image_url = request.POST.get('input_image')
            mask_image_url = request.POST.get('mask_image')
            userid = request.POST.get('userid')

            if not input_image_url or not mask_image_url or not userid:
                return JsonResponse({'status': 400, 'message': 'Missing required fields'}, safe=False)

            img = url_to_image(input_image_url)
            if img is None:
                return JsonResponse({'status': 400, 'message': 'Failed to download or read input image'}, safe=False)

            mask = url_to_image(mask_image_url, gray=True)
            if mask is None:
                return JsonResponse({'status': 400, 'message': 'Failed to download or read mask image'}, safe=False)

            # Apply inpainting using cv2
            inpainted_img = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            # Convert the inpainted image to base64
            _, buffer = cv2.imencode('.png', inpainted_img)
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
