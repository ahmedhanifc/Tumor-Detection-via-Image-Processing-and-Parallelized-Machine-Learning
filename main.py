from src.data_loader import read_images
from src.functions import *
from src.sequential import *
from src.parallel import apply_filters_parallel, process_images_threading, model_training_thread
from src.performanceMetrics import *
import glob
import pickle


if __name__ == "__main__":

    save_count = 1

# Define the path to the dataset
    dataset_path = './data/brain_tumor_dataset/'

    # List all image files in the 'yes' and 'no' directories
    yes_images = glob.glob(dataset_path + 'yes/*.jpg')
    no_images = glob.glob(dataset_path + 'no/*.jpg')

    yes_images = read_images(yes_images)
    no_images = read_images(no_images)

    print(f"Number of 'yes' images: {len(yes_images)}")
    print(f"Number of 'no' images: {len(no_images)}")

    #image = yes_images[0]
    #filtered_images = apply_filter(image)
    #display_image(filtered_images,"./plots/test_one")
    # Testing on Three Images
    yes_test =yes_images[:20]
    no_test = no_images[:20]

    
    sequential_time_yes = process_images_sequentially(yes_test)
    sequential_time_no = process_images_sequentially(no_test)
    total_sequential_time = sequential_time_yes + sequential_time_no
    print("Total Sequential Execution Time is: ", total_sequential_time)    

    results1, time_yes_test = apply_filters_parallel(yes_test)

    results2, time_no_test = apply_filters_parallel(no_test)
    

    total_threading_time = []
    non_parallel_time = []
    f = []
    P = []
    speedup = []
    efficiency = []
    amdahl_upper_limit = []
    amdahl = []
    gustafson = []

    def printMetrics(key, NUM_PROCESSES):
        f.append(non_parallel_time[key] / total_sequential_time)
        P.append(1 - f[key])
        print("Total Threading Execution Time is: ", total_threading_time[key])
        
        speedup.append(calculateSpeedup(total_sequential_time, total_threading_time[key]))
        efficiency.append(calculateEfficiency(speedup[key], NUM_PROCESSES))
        amdahl_upper_limit.append(calculateAmdhalUpperLimit(P[key]))
        amdahl.append(calculateAmdhal(P[key],NUM_PROCESSES))
        gustafson.append(calculateGustafson(P[key],NUM_PROCESSES))
        interpretMetrics(NUM_PROCESSES, speedup[key], efficiency[key], P[key], amdahl_upper_limit[key], amdahl[key], gustafson[key])

    total_threading_time.append(time_yes_test["total_time"] + time_no_test["total_time"])
    non_parallel_time.append(time_yes_test["non_parallel_time"] + time_no_test["non_parallel_time"])
    print("-"*40)
    print("Metrics for Applying Filters")
    key = 0
    NUM_PROCESSES = 6
    printMetrics(key, NUM_PROCESSES)


    write_dict_to_pickle(f"./data/filtered_images/filtered_yes_images_{save_count}", results1)
    write_dict_to_pickle(f"./data/filtered_images/filtered_no_images_{save_count}", results2)

        
    # Part Two
    
    glcm_results_1, time_glcm_results_1 = process_images_threading(results1,1)
    glcm_results_2, time_glcm_results_2 = process_images_threading(results2,0)

    total_threading_time.append(time_glcm_results_1["total_time"] + time_glcm_results_2["total_time"])
    non_parallel_time.append(time_glcm_results_1["non_parallel_time"] + time_glcm_results_2["non_parallel_time"])

    aggregated_dicts_result = aggregate_dicts(glcm_results_1,glcm_results_2)
    df = create_and_shuffle_df(aggregated_dicts_result)

    key +=1
    NUM_PROCESSES = 5
    print("-"*40)
    print("Metrics for Calculating GLCM")

    printMetrics(key, NUM_PROCESSES)
    
    key +=1


    write_dict_to_pickle(f"./data/glcm_results/yes_images_glcm_result_{save_count}",glcm_results_1)
    write_dict_to_pickle(f"./data/glcm_results/no_images_glcm_result_{save_count}",glcm_results_2)
    df.to_excel(f"./data/glcm_results/glcm_dataset_{save_count}.xlsx", index=False)


    #Part Three -- ML
    X = df.drop("Tumor", axis = 1)
    y = df["Tumor"]
    results, model_train_time = model_training_thread(X,y)

    total_threading_time.append(model_train_time["total_time"])
    non_parallel_time.append(model_train_time["non_parallel_time"])

    model_performance = convert_model_performance_to_dataframe(results)

    print("-"*40)
    print("Metrics for Calculating Model")
    key+=1
    printMetrics(key, NUM_PROCESSES)
    
    model_performance.to_excel(f"./data/model_performance_{save_count}.xlsx", index=False)

    models_from_df = model_performance["model"]
    saveModels(models_from_df,f"./data/models_{save_count}")



