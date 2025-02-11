FROM tensorflow/tensorflow

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY /src /src

CMD ["python", "src/data_loader.py"]