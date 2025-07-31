FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y python3-opencv libgl1-mesa-glx ffmpeg && \
    apt-get clean

WORKDIR /app/
COPY requirements.txt .
COPY main.py .

#RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["python", "main.py"]


