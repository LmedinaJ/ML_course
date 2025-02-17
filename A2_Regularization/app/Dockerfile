# Use Python 3.11 base image
FROM python:3.11

# Set the working directory in the container to /code/app
WORKDIR /code/app

# Copy requirements.txt into /code/app
COPY ./requirements.txt /code/app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code (including templates) into /code/app
COPY ./A2_Regularization/app /code/app

# Expose port 8000
EXPOSE 8000

# Run the app using uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
