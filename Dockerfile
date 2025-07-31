# Start from the official base image
FROM huggingface/autotrain-advanced:latest

# Run the installation commands during the build process
# This "bakes in" the dependencies so you don't download them every time.
RUN . /app/miniconda/bin/activate && pip install --upgrade autotrain-advanced

# Set the default command to run when the container starts
CMD ["sh", "-c", ". /app/miniconda/bin/activate && autotrain app --host 0.0.0.0 --port 7860"]