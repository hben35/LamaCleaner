from django.http import JsonResponse
from rest_framework.decorators import api_view
from pathlib import Path
from io import BytesIO
from PIL import Image
import requests
import numpy as np
import cv2
import base64

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

            img = url_to_image(input_image_url)
            if img is None:
                return JsonResponse({'status': 400, 'message': 'Failed to download or decode input image'}, safe=False)

            mask = url_to_image(mask_image_url)
            if mask is None:
                return JsonResponse({'status': 400, 'message': 'Failed to download or decode mask image'}, safe=False)

            # Assurer que l'image et le masque ont les mêmes dimensions
            img, mask = resize_to_same_dimension(img, mask)

            # Utiliser le modèle cv2 pour l'inpainting
            res = cv2_inpainting(img, mask)

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
    Télécharge une image depuis une URL et la convertit en un format numpy.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = np.array(Image.open(BytesIO(response.content)))

        # Convertir en niveaux de gris si demandé
        if gray and len(image.shape) == 3:  # Assurer que l'image est en couleur
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None


def resize_to_same_dimension(img, mask):
    """
    Redimensionne l'image et le masque pour avoir les mêmes dimensions.
    """
    min_height = min(img.shape[0], mask.shape[0])
    min_width = min(img.shape[1], mask.shape[1])
    img = cv2.resize(img, (min_width, min_height))
    mask = cv2.resize(mask, (min_width, min_height))
    return img, mask

def cv2_inpainting(img, mask):
    """
    Utilise le modèle cv2 pour effectuer l'inpainting.
    """
    res = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return res

def image_to_base64(image):
    """
    Convertit une image numpy en format base64.
    """
    pil_image = Image.fromarray(image)
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
