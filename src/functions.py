from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy import ndimage as nd
from skimage.filters import sobel, gabor, hessian, prewitt
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pickle
import joblib

def analyze_df(df):
    print("Dataframe Shape:", df.shape)

    print("First Five Rows:")
    print(df.head())

    print("Number of Columns:", len(df.columns.tolist()))

def aggregate_dicts(*dicts):
    results = []
    for dict in dicts:
        for value in dict.values():
            results.append(value)
    return results

def create_and_shuffle_df(data):
    df = pd.DataFrame()
    for datum in data:
        temp_df = pd.DataFrame([datum])
        df = pd.concat([df,temp_df])

    df = df.reset_index().drop("index", axis = 1).sample(frac = 1).reset_index(drop=True)
    print(len(temp_df.columns.tolist()))
    return df


def apply_entropy(image):
    filtered_image = entropy(image,disk(2))
    return filtered_image
    
def apply_gaussian_filter(image):
    filtered_image = nd.gaussian_filter(image, sigma=1)
    return filtered_image

def apply_sobel(image):
    filtered_image = sobel(image)
    return filtered_image

def apply_gabor(image):
    filtered_image =  gabor(image, frequency=0.9)[1]
    return filtered_image

def apply_hessian(image):
    filtered_image =  hessian(image, range(1,100,1))
    return filtered_image

def apply_prewitt(image):
    filtered_image =  prewitt(image)
    return filtered_image
    
def apply_filter(image):
    # Apply filters
    filtered_images = {
        'Original': image,
        'Entropy': entropy(image, disk(2)),
        'Gaussian': nd.gaussian_filter(image, sigma=1),
        'Sobel': sobel(image),
        'Gabor': gabor(image, frequency=0.9)[1],
        'Hessian': hessian(image, sigmas=range(1, 100, 1)),
        'Prewitt': prewitt(image)
    }
    return filtered_images
    
def process_images(images):
    processed_images = []
    for image in tqdm(images):
        filtered_image = apply_filter(image)
        processed_images.append(filtered_image) 
    return processed_images
    
def display_image(filtered_images,path):
    # Display each filtered image
    plt.figure(figsize=(18, 3))
    for i, (filter_name, filtered_image) in enumerate(filtered_images.items()):
            plt.subplot(1, len(filtered_images), i + 1)
            plt.imshow(filtered_image, cmap='gray')
            plt.title(filter_name)
            plt.axis('off')
    plt.savefig(f"{path}_{i}.png")
    plt.show()

def convert_model_performance_to_dataframe(results):
    data = []
    for model_name,model, metrics, params in results:
        data.append({
            'model_name': model_name,
            "model":model,
            'model_parameters': params,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1'],
            'confusion_matrix': metrics['cm']
        })
    return pd.DataFrame(data)

def write_dict_to_pickle(path,data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_dict_from_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def saveModels(models,path):
    try:
        for count,model in enumerate(models,1):
            model_name = model.__class__.__name__
            joblib.dump(model, f"{path}/{model_name}_model_{count}.joblib")
    except:
        pass

    