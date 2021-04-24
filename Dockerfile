#FROM python:3.8-slim
FROM python:3.6
WORKDIR /home/biolib
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY predict.py .
COPY data/test.zip data/
COPY features_model.csv .
COPY model.pkl .
ENTRYPOINT ["python3", "predict.py"]
