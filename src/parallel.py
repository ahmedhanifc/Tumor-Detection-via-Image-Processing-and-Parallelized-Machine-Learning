import threading as t
from src.functions import apply_entropy, apply_gaussian_filter, apply_sobel, apply_gabor, apply_hessian, apply_prewitt
import numpy as np
import skimage.feature as feature
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from src.model import train_classification_model, splitData
import itertools
import time

def model_training_thread(X, y):
    """Function to train multiple models with different parameters using threading"""
    # Define model configurations
    start = time.time()
    parallel_time = 0
    
    model_configs = {
        "RandomForest": {
            "model_class": RandomForestClassifier,
            "params": {
                "n_estimators": [10, 50, 100, 200, 300, 500],
                "max_features": ['sqrt', 'log2', None],
                "max_depth": [1, 5, 10, 20, None]
            }
        },
        "GradientBoosting": {
            "model_class": GradientBoostingClassifier,
            "params": {
                "n_estimators": [10, 50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2, 0.3],
                "max_depth": [3, 5, 10, None]
            }
        },
        "AdaBoost": {
            "model_class": AdaBoostClassifier,
            "params": {
                "n_estimators": [10, 50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.5, 1.0]
            }
        },
        "SVC": {
            "model_class": SVC,
            "params": {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "poly", "sigmoid"],
                "gamma": ["scale", "auto"]
            }
        }
    }
    '''
    model_configs = {
            "RandomForest": {
                "model_class": RandomForestClassifier,
                "params": {
                    "n_estimators": [10, 50],
                    "max_features": ['sqrt'],
                }
            },
            "GradientBoosting": {
                "model_class": GradientBoostingClassifier,
                "params": {
                    "n_estimators": [10, 50],
                    "max_depth": [3,5]
                }
        }
    }
    '''
    scores = []
    # Split the data
    X_train, X_test, y_train, y_test = splitData(X, y, test_size=0.2)
    
    # Create a print lock to avoid garbled console output
    print_lock = t.Lock()
    scores_lock = t.Lock()
    
    def process_model(model_name,model_class, params):
        """Thread function to train a model with specific parameters"""
        try:
            # Create a new model instance with the specified parameters
            model = model_class(**params)
            score = []
            
            # Train the model
            success, model, metrics = train_classification_model(model, X_train, X_test, y_train, y_test)
            with scores_lock:
                score.append(model_name)
                score.append(model)
                score.append(metrics)
                score.append(params)
                scores.append(score)

            
            # Optional debug with thread-safe printing
            if not success:
                with print_lock:
                    pass
                    #print(f"Failed to train: {model}")
        except Exception as e:
            with print_lock:
                pass
                #print(f"Error processing {model_class.__name__} with params {params}: {str(e)}")
    
    # Create threads
    threads = []
    
    # Generate all parameter combinations for each model type
    for model_name, config in model_configs.items():
        model_class = config["model_class"]
        param_names = list(config["params"].keys())
        param_values = list(config["params"].values())
        
        # Generate all combinations of parameters
        param_combinations = list(itertools.product(*param_values))
        # Create a thread for each parameter combination
        for combo in param_combinations:
            params = dict(zip(param_names, combo))
            thread = t.Thread(target=process_model, args=(model_name,model_class, params))
            threads.append(thread)
    
    # Limit concurrent threads to avoid memory issues
    max_concurrent_threads = 5  # Reduced to 5 to minimize resource contention
    thread_batch = []
    
    # Process threads in batches
    for i in range(0, len(threads), max_concurrent_threads):
        thread_batch = threads[i:i+max_concurrent_threads]
        
        parallel_start = time.time()
        # Start all threads in this batch
        for thread in thread_batch:
            thread.start()
        
        # Wait for all threads in this batch to complete
        for thread in thread_batch:
            thread.join()
        parallel_end = time.time()
        parallel_time += parallel_end - parallel_start
    
    end = time.time()
    total_time = end - start
    non_parallel_time = total_time - parallel_time
    time_dict = {
        "total_time": total_time,
        "non_parallel_time":non_parallel_time,
        "parallel_time":parallel_time
    }
    
    print("All model training complete!")
    return scores, time_dict

def process_images_threading(images, tumor_presence):
    start = time.time()
    parallel_time = 0
    results = {}
    results_lock = t.Lock()
    for image_id, filtered_image in images.items():
        threads = []
        def process_glcm_features(image_id, image, filter_name, tumor_presence):
            result = compute_glcm_features(image,filter_name)
            with results_lock:
                if image_id not in results:
                    results[image_id] = {}  # Initialize if missing
                results[image_id].update(result)  # Merge features
                results[image_id]["Tumor"] = tumor_presence

        for filter, image in filtered_image.items():
            #print(filter)
            thread = t.Thread(target = process_glcm_features, args= (image_id,image,filter, tumor_presence))
            threads.append(thread)

        max_concurrent_threads = 5
        for i in range(0,len(threads),max_concurrent_threads):
            thread_batch = threads[i:i+max_concurrent_threads]
            parallel_time_start = time.time()
            for thread in thread_batch:
                thread.start()
            for thread in thread_batch:
                thread.join()
            parallel_time_end = time.time()
            parallel_time += parallel_time_end - parallel_time_start

    end = time.time()        
    total_time = end - start
    print("Total Time with Threading:", total_time)
    non_parallel_time = total_time - parallel_time

    time_dict = {
        "total_time": total_time,
        "non_parallel_time":non_parallel_time,
        "parallel_time":parallel_time
    }
    return results, time_dict

    
def apply_filters_parallel(images):
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
    time_dict = {
        "total_time": total_time,
        "non_parallel_time":non_parallel_time,
        "parallel_time":parallel_time
    }
    return results, time_dict
        


       