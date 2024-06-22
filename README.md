# Face Detection and Recognition

Face Detection and Recognition using **DeepFace (GhostFaceNet)** model and LFW-People Dataset.

## Built With

The project is primarily built using the **Python** programming language with the help of the following libraries:

-   DeepFace
-   Pandas
-   os

## Getting Started

To start using the project, you first have to setup your local machine to meet the system prerequisites. For this, just follow the below steps:

1. Clone this repository to your local machine:
    ```sh
    git clone https://github.com/gulvaibhav20/face-recognition.git
    ```
2. Navigate to the project repository:
    ```sh
    cd face-recognition
    ```
3. Install the required Python dependencies:
    ```sh
    pip install -r requirements.txt
    ```
4. Create a sub-folder with the name `dataset`. Here, download and paste the extracted LFW - People dataset. The dataset can be accessed via this [Kaggle link](https://www.kaggle.com/datasets/atulanandjha/lfwpeople).
5. Use the `main.py` to start using the face detection and recognition model.

<br />

## Usage

1. Be ready with the image on which you wish to perform face detection and recognition process.
2. Modify the global parameters `DATASET_PATH`, `PROCESSED_DATASET_PATH`, and `input_image` according to the paths on your local machine. Here, the variables indicate the following:
    - **DATASET_PATH:** The path to the downloaded and extracted LFW - People dataset.
    - **PROCESSED_DATASET_PATH:** The path to extract and prepare the dataset for face recognition process.
    - **input_image:** The path to input image for face recognition process.
3. Run the face recognition script:
    ```sh
    python src/main.py
    ```

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
