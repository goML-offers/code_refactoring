# Use the official Python 3.11 image as the base image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Install uvicorn, a lightweight ASGI server for running FastAPI applications
RUN pip install uvicorn

# Install the required dependency python-multipart, which is used for handling file uploads in FastAPI
RUN pip install python-multipart

# Copy the entire application code to the working directory in the container
COPY . .

# Define the command to run the FastAPI application using uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
