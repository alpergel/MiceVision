FROM python:3.10-slim

EXPOSE 8501

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

RUN pip3 install -r requirements.txt
COPY . .


HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run"]
CMD ["MiceVision.py"]
