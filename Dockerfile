# Utiliser une image de python en tant que base
FROM python:3.10.7

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Créer et activer l'environnement virtuel
RUN python -m venv venv && \
    /bin/bash -c "source venv/bin/activate" && \
    pip install --upgrade pip && \
    pip install django==4.1.2 djangorestframework==3.14.0 opencv-python-headless==4.6.0.66 pytest==7.1.3 torch==1.12.1 pydantic==1.10.2 loguru==0.6.0 tqdm==4.64.1 Pillow==9.2.0 diffusers==0.4.2 transformers==4.23.1 scikit-image==0.19.3

# Télécharger le dossier lama_cleaner depuis GitHub
RUN git clone https://github.com/hben35/LamaCleaner.git && \
    mv LamaCleaner/* . && \
    rm -rf LamaCleaner

# Supprimer le fichier db.sqlite3 pour éviter les conflits de base de données
RUN rm -f db.sqlite3

# Exposer le port 8000 pour que le serveur Django puisse être accessible
EXPOSE 8000

# Commande pour exécuter le serveur Django lorsque le conteneur démarre
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
