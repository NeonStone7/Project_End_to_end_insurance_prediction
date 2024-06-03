# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application files
COPY custom_package /app/custom_package
COPY config.json /app/
COPY model_artifacts /app/model_artifacts
COPY mlruns /app/mlruns
COPY deploy /app/deploy

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r ./deploy/requirements.txt

# Expose port 2000 for the Flask app
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=./deploy/app.py

# Run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]