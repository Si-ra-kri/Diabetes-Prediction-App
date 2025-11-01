# Use Python 3.11 image
FROM python:3.11

# Set working directory inside container
WORKDIR /code

# Copy requirements first and install (to leverage caching)
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy app code
COPY ./app /code/app
COPY ./server.py /code/

# Copy the trained model into the app folder
COPY ./model.joblib /code/app/

# Expose port 8000
EXPOSE 8000

# Start FastAPI app with Uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
