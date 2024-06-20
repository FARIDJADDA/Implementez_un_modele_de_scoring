# Utilisez une image de base avec Python 3.8
FROM python:3.8-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de l'API, modèles et dépendances
COPY ./api /app/api
COPY ./models /app/models
COPY ./requirements.txt /app/requirements.txt

# Installer les dépendances
RUN pip install --no-cache-dir -r /app/requirements.txt

# Définir la commande de lancement
CMD ["python", "api/app.py"]
