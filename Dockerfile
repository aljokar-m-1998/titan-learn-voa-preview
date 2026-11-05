FROM python:3.10-slim

# Same base and cache layout as the gateway container, so what is measured here is measured
# where the adapter will actually run: HF_HOME under /code, non-root user, uvicorn on 7860.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/code/.cache/huggingface

RUN useradd -m -u 1000 user
WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user:user . .
RUN mkdir -p /code/.cache/huggingface && chown -R user:user /code

USER user
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
