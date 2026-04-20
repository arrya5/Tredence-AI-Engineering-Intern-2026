FROM python:3.10-slim

WORKDIR /app

# Install dependencies needed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy our application code
COPY . .

# Expose standard FastAPI port
EXPOSE 8000

# Run the FastAPI server by default
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
