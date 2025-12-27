FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY test_request.py .

EXPOSE 8000
EXPOSE 5000

# default command does nothing
CMD ["bash"]

