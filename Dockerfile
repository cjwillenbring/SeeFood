FROM python:3.7-slim

# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.7-slim

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install Flask gunicorn torch torchvision

RUN ls

CMD exec gunicorn --bind :$PORT --workers 1 --threads 1 app:app