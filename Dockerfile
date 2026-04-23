FROM python:3.10

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Create unprivileged user (best practice for HF Spaces)
RUN useradd -m -u 1000 user

# Install requirements globally so binaries like gunicorn are natively in the PATH
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy all files with appropriate ownership
COPY --chown=user:user . .

# Give the user ownership of the /app directory so Python can create new folders like 'data' and 'instance'
RUN chown user:user /app

# Switch to the unprivileged user before executing
USER user

EXPOSE 7860
# Run the gunicorn server
CMD ["gunicorn", "-b", "0.0.0.0:7860", "-w", "2", "--timeout", "120", "app:app"]
