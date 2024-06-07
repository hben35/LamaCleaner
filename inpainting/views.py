from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from pathlib import Path
import cv2

# Chemin vers les fichiers locaux
input_image_path = Path("/chemin/vers/le/repertoire/testimages/image1.jpg")
mask_image_path = Path("/chemin/vers/le/repertoire/testimages/mask1.jpg")

# Fonction de traitement de la requête
@api_view(['POST'])
def lamaCleaner(request):
    if request.method == "POST":
        # Utilisation des fichiers locaux au lieu de request.POST
        img = cv2.imread(str(input_image_path))
        mask = cv2.imread(str(mask_image_path), cv2.IMREAD_GRAYSCALE)

        # Votre logique de traitement ici

        # Réponse JSON de test
        response = {
            'status': 200,
            'message': "success",
            'User': "Nom_de_l'utilisateur"  # Remplacer par le nom d'utilisateur approprié
        }
        return JsonResponse(response, safe=False)
