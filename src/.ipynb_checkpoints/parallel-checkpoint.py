import threading as t
from src.functions import apply_entropy, apply_gaussian_filter, apply_sobel, apply_gabor, apply_hessian, apply_prewitt
from queue import Queue
import time 

import skimage.feature as feature
def compute_glcm_features(image, filter_name):

    image = (image * 255).astype(np.unit8)

    graycom = feature.graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

    features = {}
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        values = feature.graycoprops(graycom, prop).flatten()
        for i, value in enumerate(values):
            features[f'{filter_name}_{prop}_{i+1}'] = value
    return features


def process(images, tumor_presence):
    """
    Processes a list of images, applies all filters, computes GLCM features, and adds a "Tumor" key.

    Parameters:
    - images_list: A list of dictionaries, where each dictionary contains filtered images with keys
      representing the filter names.
    - tumor_presence: An integer (0 or 1) indicating the presence (1) or absence (0) of a tumor.

    Returns:
    - glcm_features_list: A list of dictionaries, where each dictionary contains the GLCM features for
      all filtered images of one original image and a "Tumor" key indicating the presence or absence
      of a tumor.

    Notes:
    - The function iterates over each image in the input list. For each image, it applies all filters
      and computes the GLCM features using the compute_glcm_features function.
    - The "Tumor" key is added to each dictionary to indicate whether the image is from the "yes" (tumor)
      or "no" (no tumor) list.
    - The resulting list of dictionaries can be used to create a pandas DataFrame for machine learning
      tasks.
    """
    glcm_features_list = []
    for filtered_image in images:
        glcm_features = {}
        for key,image in filtered_image.items():
            glcm_features.update(compute_glcm_features(image,key))
        glcm_features["Tumor"] = tumor_presence
        glcm_features_list.append(glcm_features)

    return glcm_features_list
    
def threading_main(images):
    # Mapping filter names to functions
    start = time.time()

    filter_functions = {
        'Entropy': apply_entropy,
        'Gaussian': apply_gaussian_filter,
        'Sobel': apply_sobel,
        'Gabor': apply_gabor,
        'Hessian': apply_hessian,
        'Prewitt': apply_prewitt,
    }
    results = {}
    results_lock = t.Lock()

    parallel_time = 0

    for image_id, image in enumerate(images):
        threads = []
        with results_lock:
            results[image_id] = {"Original": image}

        def process_filter(filter_name, filter_function, image_id, image):
            result = filter_function(image)
            with results_lock:
                results[image_id][filter_name] = result
            
        for filter_name, filter_function in filter_functions.items():
            thread = t.Thread(target = process_filter, args = (filter_name,filter_function, image_id, image))
            threads.append(thread)

        parallel_start = time.time()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        parallel_end = time.time()
        parallel_time += parallel_end - parallel_start

    end = time.time()
    total_time = end - start
    non_parallel_time = total_time - parallel_time
    return results, total_time, non_parallel_time
        


       