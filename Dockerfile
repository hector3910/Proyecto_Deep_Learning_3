FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para TensorFlow y NLTK
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias Python primero (aprovecha caché de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descargar recursos de NLTK
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

# Copiar el resto del proyecto
COPY . .

# Exponer puerto de la API
EXPOSE 8000

# Arrancar la API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
