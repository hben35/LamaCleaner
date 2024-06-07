from django.http import JsonResponse
from rest_framework.decorators import api_view
import cv2
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler

@api_view(['POST',])
def lamaCleaner(request):
    if request.method == "POST":
        input_image = request.FILES['input_image']
        mask_image = request.FILES['mask_image']
        userid = request.POST['userid']

        imagename = f"inpaint_{userid}.png"
        model = ModelManager(name="lama", device="cpu")
        img = cv2.imread(input_image)  # Utilisez input_image comme chemin vers le fichier
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        mask = cv2.imread(mask_image, cv2.IMREAD_GRAYSCALE)  # Utilisez mask_image comme chemin vers le fichier

        res = model(img, mask, get_config(HDStrategy.RESIZE))  # Vous pouvez utiliser trois types de stratégies (CROP, RESIZE, ORIGINAL)
        save_dir = "chemin/vers/votre/repertoire/de/sauvegarde"  # Assurez-vous que ce chemin existe
        cv2.imwrite(
            f"{save_dir}/{imagename}",
            res,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100, int(cv2.IMWRITE_PNG_COMPRESSION), 0],
        )

        response = {
            'status': 200,
            'message': "success",
            'User': imagename
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
