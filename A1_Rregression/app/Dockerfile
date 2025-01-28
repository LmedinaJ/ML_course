# Use Python 3.11 base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /code

# Copy requirements.txt into the container
COPY ./requirements.txt /code/requirements.txt


# Install Python dependencies
RUN pip install --no-cache-dir -r /code/requirements.txt


# Copy the application code into the container
COPY ./A1_Rregression/app /code/app

# Expose port 8000 for the app
EXPOSE 8000

# Run the app using uvicorn
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]



