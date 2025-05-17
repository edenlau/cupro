FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8081

# By default environment variables can be provided via an external .env file
# when starting the container.
CMD ["python", "src/main_web.py"]

