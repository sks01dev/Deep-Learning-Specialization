# 🦁 Deep L-Layer Neural Network for Cat/Non-Cat Classification

## ✍️ Description

This project marks the culmination of the foundational Deep Learning concepts by applying a manually built, fully **vectorized L-Layer Neural Network (DNN)** to a binary image classification task (Cat vs. Non-Cat). It integrates all the modular functions (initialization, forward/backward propagation, and parameter update) developed previously into cohesive, trainable models.

The primary goal is to demonstrate that increasing network depth (from 2 layers to 4 layers) significantly improves the model's ability to learn complex features necessary for image classification.

### Architecture Comparison:

1.  **2-Layer NN:** INPUT -> LINEAR -> ReLU -> LINEAR -> Sigmoid -> OUTPUT
2.  **L-Layer NN (4-Layers):** [LINEAR -> ReLU] $\times$ 3 -> LINEAR -> Sigmoid -> OUTPUT

---

## ⚙️ Installation

To run this notebook, you need a Python environment with the standard scientific and image processing libraries.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sks01dev/Deep-Learning-Specialization
    cd Deep-Learning-Specialization
    ```

2.  **Install dependencies:**
    (Note: The required utility files `dnn_app_utils_v3.py` and `public_tests.py` must be present alongside the notebook.)
    ```bash
    pip install numpy matplotlib h5py scipy Pillow
    ```

---

## 🏃 Usage

Execute the Jupyter Notebook cell-by-cell.

1.  **Data Preprocessing:** Images are loaded (64x64x3), flattened into vectors ($\approx 12,288$ features), and standardized by dividing by 255.
2.  **2-Layer Model Training:** The network `[12288, 7, 1]` is trained for 2500 iterations.
    * **Result:** Test Accuracy: **72%**
3.  **L-Layer Model Training:** The deep network `[12288, 20, 7, 5, 1]` (4 total layers) is trained for 2500 iterations.
    * **Result:** Test Accuracy: **80%**
4.  **Mislabelled Analysis:** The notebook plots examples where the 4-layer model fails, showing common challenges like unusual cat positions, similar background colors, and brightness variations.

---

## 🛠️ Technologies Used

| Technology | Purpose | Badge/Icon |
| :--- | :--- | :--- |
| **Python** | Core implementation language | [![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/doc/) |
| **NumPy** | Vectorized matrix operations and calculation engine | [![NumPy](https://img.shields.io/badge/NumPy-1.x-blue?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/doc/) |
| **Matplotlib** | Plotting costs and visualizing misclassified images | [![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-red?style=flat-square&logo=matplotlib&logoColor=white)](https://matplotlib.org/stable/contents.html) |
| **h5py** | Interacting with the `.h5` dataset file | [![h5py](https://img.shields.io/badge/h5py-3.x-blueviolet?style=flat-square)](https://docs.h5py.org/en/latest/) |
| **PIL/Scipy** | Image handling and model testing with external images | [![Pillow](https://img.shields.io/badge/Pillow-10.x-00628D?style=flat-square&logo=python&logoColor=white)](https://pillow.readthedocs.io/en/stable/index.html) |

---

## 🧠 Key Learnings

1.  **The Depth Advantage:** Successfully proved that the 4-layer DNN achieved significantly better performance (80% accuracy) than both the Logistic Regression (70%) and the 2-layer NN (72%), confirming the power of deep architectures to extract hierarchical features.
2.  **Full Pipeline Integration:** Mastered the Deep Learning workflow by integrating complex, low-level components (`initialize_parameters_deep`, `L_model_forward`, `L_model_backward`) into a single, cohesive training loop.
3.  **Overfitting Awareness:** Observed the high training accuracy (98.5%) on the L-Layer model compared to the test accuracy (80%), highlighting the challenge of **overfitting** and suggesting the future need for regularization and early stopping.

---

## 📚 References

* *Andrew Ng's Deep Learning Specialization - Course 1, Week 4*
