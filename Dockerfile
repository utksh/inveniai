# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /INVENAI

# Copy the current directory contents into the container at /app
COPY . /INVENAI

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirement.txt
RUN pip install pandas


# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python", "app.py"]
