from django.http import JsonResponse
from rest_framework.decorators import api_view

import cv2
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler

# Create your views here.

@api_view(['POST',])
def lamaCleaner(request):
    if request.method == "POST":
        # Vérifier si les fichiers sont présents dans la demande
        if 'input_image' in request.FILES and 'input_mask' in request.FILES:
            input_image_file = request.FILES['input_image']
            mask_image_file = request.FILES['input_mask']

            # Autres données de la requête
            userid = request.POST.get('userid', '')

            imagename = f"inpaint_{userid}.png"
            model = ModelManager(name="lama", device="cpu")
            img = cv2.imdecode(input_image_file.read(), cv2.IMREAD_COLOR)
            mask = cv2.imdecode(mask_image_file.read(), cv2.IMREAD_GRAYSCALE)

            res = model(img, mask, get_config(HDStrategy.RESIZE))  # vous pouvez utiliser trois types de stratégies (CROP, RESIZE, ORIGINAL)
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
        else:
            # Si les fichiers sont manquants, retourner une réponse d'erreur
            response = {
                'status': 400,
                'error': 'Les fichiers input_image et input_mask sont requis.'
            }
            return JsonResponse(response, status=400)
    else:
        # Gérer les autres méthodes HTTP si nécessaire
        response = {
            'status': 405,
            'error': 'Méthode non autorisée.'
        }
        return JsonResponse(response, status=405)

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
