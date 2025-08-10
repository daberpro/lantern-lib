# Lantern
**A Lightweight Neural Network Library in C++ — Built from Scratch**
> From-scratch C++ neural network library with custom autograd, ArrayFire acceleration, and a focus on transparency.

![Test Build](https://github.com/daberpro/lantern-lib/actions/workflows/cmake-single-platform.yml/badge.svg)
![GitHub release](https://img.shields.io/github/v/release/daberpro/lantern-lib?include_prereleases)
![GitHub](https://img.shields.io/github/license/daberpro/lantern-lib)
![Code Size](https://img.shields.io/github/languages/code-size/daberpro/lantern-lib)
![Language](https://img.shields.io/github/languages/top/daberpro/lantern-lib)
![GitHub issues](https://img.shields.io/github/issues/daberpro/lantern-lib)
![GitHub pull requests](https://img.shields.io/github/issues-pr/daberpro/lantern-lib)
![GitHub Repo stars](https://img.shields.io/github/stars/daberpro/lantern-lib)
![GitHub forks](https://img.shields.io/github/forks/daberpro/lantern-lib)


Lantern is a deep learning library written entirely in **C++**, built on top of **ArrayFire** for fast tensor operations.  
The goal is to create a fully transparent neural network framework where every detail — from weight initialization to backpropagation — is visible and customizable.  

⚠️ **Warning**: `latern-lib` is still under development and currently has limited features.  
If you have suggestions or improvements, feel free to contact me:  
📧 **Email**: `daber.coding@gmail.com`  
💬 **Discord**: `daberdev`  

---

## ✨ Features  
- Feedforward neural networks (Perceptron-based)  
- Reverse-mode automatic differentiation (custom autograd engine)  
- Multiple optimizers:  
  - Gradient Descent (GD)  
  - RMSProp  
  - AdaGrad  
  - Adam (Adaptive Moment Estimation)  
- Activations: Swish, Sigmoid, Softmax (full Jacobian)  
- Loss functions: Sum Squared Residual, Cross Entropy  
- Weight initialization: Normal Distribution + Xavier/Glorot initialization  
- Mini-batch and single-sample training  
- Explicit math — no “magic box” code  

---

## 🛠 Getting Started  

### Dependencies  
To build Lantern from source, you will need:  
- [ArrayFire](https://arrayfire.com/)  
- [Matplot++](https://alandefreitas.github.io/matplotplusplus/)  
- [HDF5](https://www.hdfgroup.org/) (with C++ enabled)  

Lantern uses **CMake** for building. Development is done on **Windows** using **Visual Studio 2022**, but the code should be portable.  
Minimum CMake version: **4.0**  

---

## 🧪 Example: XOR Training  
Here’s a minimal example to train a neural network on the XOR problem:  

```cpp
#include "../pch.h"
#include "../Headers/Logging.h"
#include "../FeedForwardNetwork/FeedForwardNetwork.h"

int main() {
    af::setSeed(static_cast<uint64_t>(std::time(nullptr)));

    // XOR input and target data
    double input_data[] = {
        1.0, 1.0, 0.0, 0.0,
        1.0, 0.0, 1.0, 0.0
    };

    double target_data[] = {
        0.0, 1.0, 1.0, 0.0
    };

    af::array input = af::array(4, 2, input_data);
    af::array target = af::array(4, 1, target_data);

    // Define network architecture
    lantern::ffn::layer::Layer layer;
    layer.Add<lantern::ffn::node::NodeType::NOTHING>(2);
    layer.Add<lantern::ffn::node::NodeType::SWISH>(4);
    layer.Add<lantern::ffn::node::NodeType::SWISH>(6);
    layer.Add<lantern::ffn::node::NodeType::SIGMOID>(1);

    // Optimizer
    lantern::ffn::optimizer::AdaptiveMomentEstimation adam;

    // Create and train model
    lantern::feedforward::FeedForwardNetwork model(
        &input,   // Pointer to training input data
        &target,  // Pointer to training target data
        &layer,   // Pointer to the defined layer structure
        {4},      // Vector of class index boundaries (only 1 group of 4 samples here)
        1e-08,    // Learning rate
        200       // Number of epochs
    );

    model.Train<4>(
        adam,
        lantern::loss::SumSquareResidual,
        lantern::derivative::SumSquareResidual,
        lantern::activation::Linear
    );

    // Prediction
    std::cout << "Prediction Result:\n";
    af::array result;
    model.Predict(
        input,
        result,
        lantern::activation::Linear
    );
    std::cout << result << '\n';

    return EXIT_SUCCESS;
}
````

---

## 📖 Line-by-Line Explanation

1. **Includes**

   * `pch.h` — Precompiled headers to speed up build times
   * `Logging.h` — Lantern’s logging utility
   * `FeedForwardNetwork.h` — The main neural network class

2. **Set Random Seed**

   ```cpp
   af::setSeed(static_cast<uint64_t>(std::time(nullptr)));
   ```

   Ensures randomness in weight initialization, changing each run.

3. **Prepare Training Data**

   * `input_data`: Each column is a feature, each row is a sample
   * `target_data`: Expected outputs for each input

4. **Convert to ArrayFire arrays**

   * `af::array input` — Shape `(4, 2)` → 4 samples × 2 features
   * `af::array target` — Shape `(4, 1)` → 4 samples × 1 label

5. **Build Network Layers**

   ```cpp
   layer.Add<lantern::ffn::node::NodeType::NOTHING>(2); // Input layer
   layer.Add<lantern::ffn::node::NodeType::SWISH>(4);   // Hidden layer 1
   layer.Add<lantern::ffn::node::NodeType::SWISH>(6);   // Hidden layer 2
   layer.Add<lantern::ffn::node::NodeType::SIGMOID>(1); // Output layer
   ```

   Defines layer sizes and activation functions.

6. **Choose Optimizer**

   ```cpp
   lantern::ffn::optimizer::AdaptiveMomentEstimation adam;
   ```

7. **Create Model**

   ```cpp
   lantern::feedforward::FeedForwardNetwork model(
       &input, &target, &layer,
       {4},      // Class index boundaries
       1e-08,    // Learning rate
       200       // Epochs
   );
   ```

   `{4}` means: “Class 0 samples end at index 4.”
   If you had: `{15, 31}`, it means:

   * Class 0 ends at index 15
   * Class 1 ends at index 31

8. **Train the Model**

   ```cpp
   model.Train<4>(
       adam,
       lantern::loss::SumSquareResidual,
       lantern::derivative::SumSquareResidual,
       lantern::activation::Linear
   );
   ```

   `<4>` is the batch size for training.

9. **Predict & Print Results**

   ```cpp
   model.Predict(input, result, lantern::activation::Linear);
   std::cout << result << '\n';
   ```

---

