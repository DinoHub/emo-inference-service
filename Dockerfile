# TODO: replace FROM
FROM dleongsh/espnet:202302-torch1.12-cu113-runtime-demo
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV TRANSFORMERS_CACHE="/demo/models/transformers_cache"
ENV TORCH_HOME='/demo/models'

CMD ["python", "src/app.py"]
