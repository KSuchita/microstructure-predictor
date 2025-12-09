# Stage 1: Build Stage (to run model_setup.py and generate joblib files)
FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_setup.py .
COPY cleaned_dataset.csv .

RUN python model_setup.py

# Stage 2: Production Stage
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files and folders directly into the working directory
# This assumes your local directory looks like: app.py, static/, templates/, Dockerfile, etc.
COPY . .  
# Note: The model artifacts will be overwritten by this step, 
# so we must ensure we copy them AFTER the general files.
# To be safest, let's keep the explicit copy commands and just add the crucial one:

# Copy application code and assets
COPY app.py .
COPY static static
COPY templates templates
# -------------------------------

# Copy the generated model artifacts from the builder stage
COPY --from=builder /app/*.joblib .

EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]