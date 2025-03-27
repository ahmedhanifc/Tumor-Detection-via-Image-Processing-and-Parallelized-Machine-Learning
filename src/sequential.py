import time
from src.functions import process_images

def process_images_sequentially(images):
    start_time = time.time()
    processed = process_images(images)
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time