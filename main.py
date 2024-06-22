import os
import glob
import shutil
import pandas as pd
from deepface import DeepFace


# Global Variables
DATASET_PATH = (
    "dataset/lfw_funneled"  # Location of the LFW People dataset on your local system.
)
PROCESSED_DATASET_PATH = (
    "processed_dataset"  # Location of the pre-processed dataset on your local system.
)


def preprocess_dataset():
    """
    Pre-processing the dataset by fetching all the images across all the sub-folders and
    placing them under a common directory path.
    """
    print("Starting pre-processing of the LFW People dataset")
    try:
        os.mkdir(PROCESSED_DATASET_PATH)
        for folder in os.listdir(DATASET_PATH):
            src_dir = os.path.join(DATASET_PATH, folder)
            for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
                shutil.copy(jpgfile, PROCESSED_DATASET_PATH)
        print("Sucessfully prepared the pre-processed dataset for Image recognition")
    except Exception as err:
        print(f"ERROR: Pre-processing failed with error: {err}")
        return False
    return True


def recognize_face(
    input_path, db_path=PROCESSED_DATASET_PATH, model_name="GhostFaceNet"
):
    """
    Main Function to fetch the input 2D image, detect face from it and recognize it
    from the available LFW People dataset.
    """
    try:
        print("Finding relevant matches from the dataset")
        output = DeepFace.find(
            img_path=input_path, db_path=db_path, model_name=model_name
        )
        result_df = pd.concat(output)
        if result_df.empty:
            print("Unable to recognize face from the existing dataset")
        else:
            print("Successfully completed Face recognition process")
        return result_df
    except Exception as err:
        print(f"ERROR: Face recognition process failed with error: {err}")
        return pd.DataFrame()


def get_detected_face(deep_face_output):
    """
    Utility Function to fetch the detected face from the deepface model output
    """
    detected_face_list = deep_face_output["identity"].apply(
        lambda x: " ".join(os.path.basename(x)[:-4].split("_")[:-1])
    )
    if len(detected_face_list.unique()) == 1:
        return detected_face_list[0]
    else:
        occurence_dict = {}
        for name in detected_face_list:
            if occurence_dict.get(name) is None:
                occurence_dict[name] = 1
            else:
                occurence_dict[name] += 1
        return sorted(occurence_dict, key=occurence_dict.get, reverse=True)[0]  # type: ignore


if __name__ == "__main__":
    # Perform preprocessing of dataset if processed-dataset doesn't exist
    preprocessing = True
    if os.path.isdir(PROCESSED_DATASET_PATH):
        preprocessing = False

    if preprocessing:
        preprocess_dataset()

    input_image = "sample.jpg"
    deep_face_output = recognize_face(input_image)
    if not deep_face_output.empty:
        face = get_detected_face(deep_face_output)
        print(f"Detected and Recognized Face from the input: {face}")
    else:
        print("Unable to detect and recognize the Face")
