# README
## MMWave-Radar-AI-Project

---
This README explains the work completed in **Part 1 (Radar Simulation)**, **Part 2 (Model Training)**, and **Part 3 (Clutter-Resistant Evaluation + Synthetic Scene Generation)** of the Radar-Based Metal Detection Pipeline.

---

# **Part 1 — Synthetic Radar Signal Processing**

### **Objective**

Simulate realistic radar-like signals (1D fast-time FMCW samples and 2D range–Doppler maps) for:

* Empty-room scenarios
* Metal-object targets
* Clutter environments

### **What Was Implemented**

✔ FMCW radar waveform simulation
✔ Range and Doppler FFT
✔ Range–Doppler heatmap generation
✔ Clutter and multi-target scenarios
✔ Noise injection (SNR variations)
✔ Saving heatmaps and raw IQ data

### **Key Features**

* Uses **24 GHz FMCW model**
* Adds **fractional-delay echo simulation** for accurate range modelling
* Implements **Doppler phase shift** across slow-time pulses
* Generates **2D heatmaps** using `fftshift(rangeFFT -> dopplerFFT)`
* Class-balanced dataset for:

  * **metal**
  * **nonmetal**

### **Primary Output**

* `radar_simulation.ipynb`
* Folder structure:

  ```
  artifact/
    heatmaps/          # range-doppler images (128×128)
    radar_signals/     # raw IQ frames (.npy)
    labels.json        # metadata for every sample
  ```

---

# **Part 2 — CNN Model Training for Metal vs Non-Metal**

### **Objective**

Train a deep-learning classifier on the synthetic radar heatmaps.

### **Pipeline**

1. **Load dataset** from `artifact/heatmaps/`
2. **Preprocess images**

   * Resize → 128×128
   * Normalize → [0,1]
3. **Build CNN (MetalCNN)**

   * 4 convolution blocks
   * 128-dim fully connected layer
   * Softmax output (2 classes)
4. **Train/Val split (80/20)**
5. **Compute metrics**:

   * Accuracy
   * Precision/Recall
   * Confusion Matrix

### **Key Features**

✔ Stable architecture (no shape mismatch)
✔ Saved model → `artifact/models/metal_cnn.pth`
✔ Visualizations:

* Training curves
* Sample predictions
* Confusion matrix

---

# **Part 3 — Clutter-Robust Synthetic Scene Evaluation**

### **Objective**

Stress-test the trained CNN against **complex cluttered scenes** not seen in training.

### **What Was Implemented**

✔ **Advanced clutter generator**

* Random shapes
* Noise
* Gaussian blur
* Synthetic occlusion over metal heatspot

✔ **Background subtraction pipeline**

* Gaussian blur
* Median filter
* Morphological opening
* Foreground extraction mask

✔ **Scene classification**

* Convert preprocessed scene to tensor
* Run inference using trained CNN
* Track accuracy across 200–300 random scenes

### **Additional Visualizations**

✔ **Misclassification viewer**
✔ **Correct-classification viewer**
✔ **Side-by-side comparison:**

* Background
* Cluttered image
* Extracted foreground
* Model prediction

---

# **Summary of Deliverables**

### **Part 1**

* `radar_simulation.ipynb`
* `artifact/heatmaps/*.png`
* `artifact/radar_signals/*.npy`
* `artifact/labels.json`

### **Part 2**

* `metal_cnn_training.ipynb`
* `artifact/models/metal_cnn.pth`
* `training_curves.png`
* `confusion_matrix.png`

### **Part 3**

* `clutter_evaluation.ipynb`
* `clutter_visualizations/`

  * misclassified samples
  * correctly classified samples
  * clutter-vs-clean comparisons

---

# **Future Extensions**

* Integrate **real radar sensor data**
* Real-time radar pipeline (Part 4)
* Edge deployment optimizations
* Transformer-based classifier
* Advanced spatial–temporal filtering


