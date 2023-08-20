# Base image
#FROM python:3.9-slim-buster
FROM python:3.8-slim
# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the application code to the working directory
COPY . .

# Set the command to run the Flask appdocker build -t cnn-app .

CMD ["python", "app.py"]