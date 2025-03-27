from src.data_loader import read_images
from src.functions import *
from src.sequential import *
from src.parallel import apply_filters, process_images_threading, process_images_sequential
from src.performanceMetrics import *
import pandas as pd
import glob
import pickle


if __name__ == "__main__":
    # Define the path to the dataset
    dataset_path = './data/brain_tumor_dataset/'
    
    # List all image files in the 'yes' and 'no' directories
    yes_images = glob.glob(dataset_path + 'yes/*.jpg')
    no_images = glob.glob(dataset_path + 'no/*.jpg')
    
    yes_images = read_images(yes_images)
    no_images = read_images(no_images)
    
    # print(f"Number of 'yes' images: {len(yes_images)}")
    # print(f"Number of 'no' images: {len(no_images)}")

    #image = yes_images[0]
    #filtered_images = apply_filter(image)
    #display_image(filtered_images,"./plots/test_one")
    # Testing on Three Images
    yes_test =yes_images[:2]
    no_test = no_images[:2]
    
    # sequential_time_yes = process_images_sequentially(yes_test)
    # sequential_time_no = process_images_sequentially(no_test)
    # total_sequential_time = sequential_time_yes + sequential_time_no
    # print("Total Sequential Execution Time is: ", total_sequential_time)    

    #results1, total_time1, non_parallel_time1 = apply_filters(yes_test)

    # results2, total_time2, non_parallel_time2 = threading_main(no_test)

    '''
    total_threading_time = total_time1 + total_time2
    non_parallel_time = non_parallel_time1 + non_parallel_time2 
    
    f = non_parallel_time / total_sequential_time
    P = 1 - f
    print("Total Threading Execution Time is: ", total_threading_time)

    speedup = calculateSpeedup(total_sequential_time, total_threading_time)
    efficiency = calculateEfficiency(speedup, 6)
    amdahl_upper_limit = calculateAmdhalUpperLimit(P)
    amdahl = calculateAmdhal(P,6)
    gustafson = calculateGustafson(P,6)

    interpretMetrics(6, speedup, efficiency, P, amdahl_upper_limit, amdahl, gustafson)
    '''

    #%%
    with open("./data/image_filters.pkl", "rb") as f:
        results1 = pickle.load(f)    
    #%%
    glcm_results = process_images_threading(results1,1)
    df = create_and_shuffle_df(glcm_results)
    

    analyze_df(df)



    


