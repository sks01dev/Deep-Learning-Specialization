# 🌸 Planar Data Classification with a Single Hidden Layer Neural Network

## ✍️ Description

This project implements a **two-class classification Neural Network (NN) with a single hidden layer** to solve a non-linearly separable classification problem. The core objective is to manually build the foundational components of a neural network—including forward propagation, computing the loss, backpropagation, and parameter updates—using only NumPy.

The project highlights the significant advantage of a neural network's ability to model complex, non-linear decision boundaries compared to a simple Logistic Regression classifier on the same dataset.

### Project Goals:
* Implement a 2-layer Neural Network (input layer, one hidden layer, output layer).
* Use the **$\tanh$ (hyperbolic tangent)** as the non-linear activation function in the hidden layer.
* Implement the **Cross-Entropy Loss** function.
* Implement **Vectorized Forward and Backward Propagation** from scratch.
* Compare the performance against Logistic Regression on a non-linearly separable dataset.

---

## ⚙️ Installation

To run this notebook locally, you need a Python environment with the following dependencies.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy scikit-learn matplotlib
    ```

---

## 🏃 Usage

The project is structured as a Jupyter Notebook (`Planar_data_classification_with_one_hidden_layer.ipynb`).

1.  **Start Jupyter:**
    ```bash
    jupyter notebook
    ```
2.  **Open and run the notebook.**

The notebook executes the following steps:
1.  **Load and Visualize the "Flower" Dataset**: A non-linearly separable dataset shaped like a flower (red and blue points).
2.  **Logistic Regression Test**: A baseline test using `scikit-learn`'s Logistic Regression, which yields poor accuracy ($\approx 47\%$).
3.  **NN Model Construction**: Step-by-step implementation of the 2-layer NN functions:
    * `layer_sizes()`
    * `initialize_parameters()`
    * `forward_propagation()`
    * `compute_cost()`
    * `backward_propagation()`
    * `update_parameters()`
4.  **NN Training and Evaluation**: The model is trained using **Gradient Descent** over 10,000 iterations, resulting in high accuracy ($\approx 90\%$).
5.  **Hyperparameter Tuning**: Optional exercise exploring how varying the hidden layer size ($n_h$) impacts performance and demonstrates overfitting.

---

## 🛠️ Technologies Used

| Technology | Purpose | Badge/Icon |
| :--- | :--- | :--- |
| **Python** | Primary programming language | [![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/doc/) |
| **NumPy** | Core library for vectorized matrix operations | [![NumPy](https://img.shields.io/badge/NumPy-1.x-blue?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/doc/) |
| **Scikit-learn** | Used for the baseline Logistic Regression model | [![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24-orange?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/documentation.html) |
| **Matplotlib** | Data and decision boundary visualization | [![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-red?style=flat-square&logo=matplotlib&logoColor=white)](https://matplotlib.org/stable/contents.html) |
| **Jupyter** | Notebook environment for development and visualization | [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/documentation) |

---

## 📊 Dataset Used

* **Planar Data ("Flower" Dataset)**: A synthetic, 2-class (Red/Blue) dataset specifically generated to be **non-linearly separable**.
* **Features (X)**: 2 features (x1, x2), shape $(2, 400)$.
* **Labels (Y)**: 1 label (0 or 1), shape $(1, 400)$.

---

## 🧠 Key Learnings

1.  **NN vs. Linear Models**: Clearly demonstrated that a Neural Network can successfully model a complex "flower" shaped decision boundary (Accuracy: $\approx 90\%$) where Logistic Regression fails (Accuracy: $\approx 47\%$).
2.  **Backpropagation and Calculus**: Gained a deep, hands-on understanding of implementing the **backpropagation algorithm** by translating calculus-derived equations into vectorized NumPy code (e.g., computing $dZ^{[2]}$, $dW^{[2]}$, $dZ^{[1]}$, etc.).
3.  **Non-linear Activation ($\tanh$)**: Confirmed the importance of a non-linear activation function in the hidden layer (here, $\tanh$) to move beyond linear decision boundaries.
4.  **Vectorization**: Achieved efficient computation of all steps (Forward/Backward Prop) across all $m$ examples by using vectorized NumPy operations, avoiding explicit loops.

---

## ✨ Results

| Model | Decision Boundary | Training Accuracy |
| :--- | :--- | :--- |
| **Logistic Regression** | Linear | $\approx 47\%$ |
| **1-Hidden Layer NN** ($n_h=4$) | Non-linear ("Flower" shape) | $\approx 90\%$ |

The single hidden layer Neural Network successfully learned the intricate patterns in the data, demonstrating its capacity for complex pattern recognition.

---

## 🚀 Future Work

* **Implement Regularization (L2/Dropout)**: To allow the use of very large hidden layers (e.g., $n_h=50$) without overfitting the training data, as suggested in the interpretation section.
* **Explore Different Activations**: Replace the $\tanh$ activation with **ReLU** or **Sigmoid** to observe their impact on convergence speed and final accuracy.
* **Implement Different Optimizers**: Introduce modern optimization algorithms like **Adam** or **RMSprop** to replace standard Gradient Descent.

---

## 📚 References

* **Andrew Ng's Deep Learning Specialization - Course 1, Week 3**
* [http://cs231n.github.io/neural-networks-case-study/](http://cs231n.github.io/neural-networks-case-study/) (Referenced in the original notebook)
