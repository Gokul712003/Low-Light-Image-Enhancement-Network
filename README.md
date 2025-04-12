# C-LIENet (TensorFlow 2.x Adaptation with LOL Dataset)

This project is an adaptation of the C-LIENet (A Multi-Context Low-Light Image Enhancement Network) for compatibility with TensorFlow 2.x. It aims to enhance low-light images using a deep learning approach.

## Original Project

This work is based on the original C-LIENet implementation by Praveen Ravirathinam. You can find the original repository here:
[https://github.com/praveen-ravirathinam/C-LIENet-A-Multi-Context-Low-Light-Image-Enhancement-Network](https://github.com/praveen-ravirathinam/C-LIENet-A-Multi-Context-Low-Light-Image-Enhancement-Network?tab=readme-ov-file)

## Dataset

This adaptation utilizes the **LOL Dataset (Low-Light Paired Dataset)** for training and testing.

*   **Source:** [https://www.kaggle.com/datasets/soumikrakshit/lol-dataset](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset)
*   **Structure:** The dataset should be organized into `train` and `test` folders, each containing `high` (normal light) and `low` (low light) subfolders with paired images having the same filename.
    ```
    lol_dataset/
    ├── train/
    │   ├── high/
    │   │   ├── 1.png
    │   │   └── ...
    │   └── low/
    │       ├── 1.png
    │       └── ...
    └── test/
        ├── high/
        │   ├── 1.png
        │   └── ...
        └── low/
            ├── 1.png
            └── ...
    ```

## Modifications

The primary modifications made to the original project include:

1.  **TensorFlow 2.x Compatibility:** Updated the code (imports, optimizers, model saving/loading, metrics calculation) to work with TensorFlow 2.x and Keras API integrated within TensorFlow.
2.  **Dataset Integration:** Adapted the data loading process to use the LOL dataset structure.
3.  **Training Duration:** The model in this version was trained for a limited number of epochs (specifically 10 epochs were intended, though the provided log shows an interruption earlier). The original paper might suggest longer training for optimal results.

## Files

*   `C-LIENet_Train.ipynb`: Jupyter Notebook for training the C-LIENet model.
*   `C-LIENet_Test.ipynb`: Jupyter Notebook for testing the trained model and evaluating metrics.
*   `model/`: Folder intended to store the trained model weights (e.g., `model_clienet_best.h5`).
*   `clienet/predictions/`: Folder where predicted (enhanced) images from the test set can be saved (optional, requires uncommenting code in the test notebook).

## Setup

1.  **Clone the repository (or set up your project files).**
2.  **Download the LOL Dataset** from the Kaggle link above and place it in the project's root directory, ensuring the folder structure matches the one described under the "Dataset" section.
3.  **Create a Python environment** (e.g., using conda or venv).
4.  **Install necessary libraries:**
    ```bash
    pip install tensorflow tensorflow-gpu # Or just tensorflow if no GPU
    pip install numpy opencv-python scikit-image matplotlib scipy Pillow ipykernel jupyter
    ```
    *(Note: Ensure library versions are compatible, especially TensorFlow >= 2.0)*

## Usage

1.  **Training:**
    *   Open and run the `C-LIENet_Train.ipynb` notebook.
    *   Verify the dataset paths defined in the second code cell match your setup.
    *   Adjust `NUM_EPOCHS` and `BATCH_SIZE` if desired.
    *   The training process will save the best model weights (based on `val_psnr`) to the `model/` directory as `model_clienet_best.h5`.

2.  **Testing:**
    *   Open and run the `C-LIENet_Test.ipynb` notebook.
    *   Verify the dataset paths and the `model_name` in the second code cell match your setup and the saved weights file.
    *   The notebook will load the trained weights, predict enhanced images for the test set, and calculate evaluation metrics (PSNR, SSIM, VIFP, AB).
    *   You can uncomment the `cv2.imwrite` lines in the "CALCULATE METRICS" section to save the input and output images to the `clienet/predictions/` folder.

## Results

Running the test notebook will output the average PSNR, SSIM, VIFP, and AB metrics comparing the low-light inputs and the model's enhanced outputs against the ground truth high-light images. Keep in mind that results will depend heavily on the training duration; the limited epochs used here might not yield optimal performance compared to the original paper.

## Acknowledgements

*   Credit to **Praveen Ravirathinam** for the original C-LIENet implementation and paper.
*   Thanks to the creators of the **LOL Dataset**.

## License

*(Consider adding a license here. If the original project has a license, it's often appropriate to use the same one. Otherwise, MIT or Apache 2.0 are common choices for open-source projects.)*
