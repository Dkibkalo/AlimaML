FROM python:3.10


RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    git \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY . /app


ENV MAX_JOBS=2

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.enableCORS=false"]
