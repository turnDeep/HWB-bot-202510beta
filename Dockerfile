# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV TZ=Asia/Tokyo

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# - cron for scheduled tasks
# - procps for process management (useful for debugging)
RUN apt-get update && apt-get install -y cron procps && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Create log directory
RUN mkdir -p /app/logs

# Add the cron job
# This runs the HWB scan at 7:00 AM JST, Tuesday-Saturday (after US market close Mon-Fri)
RUN (crontab -l 2>/dev/null; echo "0 7 * * 2-6 python -m backend.hwb_scanner_cli >> /app/logs/hwb.log 2>&1") | crontab -

# Create an entrypoint script to start cron and the web server
COPY <<'EOF' /app/entrypoint.sh
#!/bin/bash
# Start the cron daemon
cron
# Start the FastAPI server
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
EOF

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Expose the port the app runs on
EXPOSE 8000

# Run the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]