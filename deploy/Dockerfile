# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the application files
COPY custom_package /app/custom_package
COPY config.json /app/
COPY model_artifacts /app/model_artifacts
COPY mlruns /app/mlruns

COPY requirements.txt /app/

# Install the dependencies
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Specify the command to run the application
CMD ["python", "app.py"]
