FROM python:3.10-slim

# Instala compiladores y librerías necesarias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && apt-get clean

# Crea carpeta de trabajo
WORKDIR /app/

# Copia dependencias y código
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .

# Puerto expuesto
EXPOSE 5000

CMD ["python", "main.py"]
