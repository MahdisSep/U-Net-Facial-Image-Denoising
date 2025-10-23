# 🎭 U-Net for Facial Image Denoising and Restoration

## 📝 Overview
This project tackles the challenging task of **Image Denoising** using a **U-Net** deep convolutional network. The goal is to accurately restore clean images from corrupted inputs (images affected by noise). The implementation focuses on facial images from the popular **FER2013** dataset (Facial Expression Recognition), demonstrating the model's ability to learn intricate spatial features necessary for high-quality restoration.

The methodology involves training the U-Net to map a **noisy image** (input) to its corresponding **clean image** (target).

## 🛠️ Technologies Used
* **Framework:** TensorFlow / Keras
* **Language:** Python
* **Libraries:** `NumPy`, `Pandas` (for CSV parsing), `OpenCV (cv2)`, `Matplotlib`, `scikit-image`.
* **Architecture:** Custom **U-Net** Implementation.

## 🎯 Key Implementation Highlights

### 1. Data Processing and Noise Injection
* **Dataset:** Utilized the FER2013 dataset, which consists of grayscale $48 \times 48$ facial expression images.
* **Custom Parser:** Developed custom functions to parse raw pixel strings from the `fer2013.csv` file and restructure them into $48 \times 48$ NumPy arrays.
* **Noise Simulation:** Introduced **Gaussian Noise** (additive white noise) to the original training images to create the noisy input data ($x_{train}$), simulating real-world image corruption. The original clean images served as the ground-truth targets ($y_{train}$).

### 2. U-Net Architecture
The U-Net model is employed due to its effectiveness in image-to-image translation tasks, especially segmentation and restoration.

* **Structure:** It follows the classic **Encoder-Decoder** structure with crucial **Skip Connections**.
    * **Encoder (Contracting Path):** Sequentially reduces image resolution via MaxPooling while increasing the number of feature channels, capturing context.
    * **Decoder (Expanding Path):** Sequentially increases image resolution via UpSampling/Conv2D Transpose, allowing for precise localization.
    * **Skip Connections:** Directly connect feature maps from the Encoder to the Decoder at the same resolution level. These connections are vital for transferring fine-grained details lost during the encoding process, which is essential for accurate pixel-level restoration.
* **Loss Function:** Trained using **Mean Squared Error (MSE)** loss, which minimizes the average squared difference between the predicted pixel values and the ground-truth clean pixel values.

### 3. Training and Evaluation
* **Training Strategy:** Implemented `EarlyStopping` (to prevent overfitting) and `ModelCheckpoint` (to save the best weights based on validation loss).
* **Optimizer:** Used the **Adam optimizer** with a low learning rate (e.g., $0.00001$) for stable convergence during deep network training.
* **Metrics:** Evaluated model performance using Loss (MSE) and **Accuracy** (as a proxy for reconstruction quality).

## 📂 Repository Structure

```

├── U-Net-Facial-Image-Denoising/
│   ├── README.md                 \<-- You are here
│   ├── Q2 (1).ipynb              \<-- Full Denoising pipeline, U-Net definition, training, and visualization
│   ├── q2.py                     \<-- Python script containing the U-Net definition and main logic
│   └── fer2013.csv               \<-- Placeholder for the dataset (must be acquired separately)

````

### showcase

![images](https://github.com/MahdisSep/U-Net-Facial-Image-Denoising/blob/main/results/results1.png)
![images](https://github.com/MahdisSep/U-Net-Facial-Image-Denoising/blob/main/results/results2.png)

-----

## ⚙️ How to Run

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/MahdisSep/U-Net-Facial-Image-Denoising.git]
    cd U-Net-Facial-Image-Denoising
    ```
2.  **Dataset Setup:** Obtain the `fer2013.csv` file (e.g., from Kaggle or other sources) and place it in the project root directory.
3.  **Install Dependencies:**
    ```bash
    pip install tensorflow keras numpy pandas matplotlib opencv-python
    ```
4.  **Execute:** Run the `Q2 (1).ipynb` notebook in a Jupyter or Colab environment. The notebook handles data loading, preprocessing, noise injection, model training, and visualization of the noisy input vs. the U-Net's denoised output.


