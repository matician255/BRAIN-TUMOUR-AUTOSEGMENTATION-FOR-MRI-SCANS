# 🧠 Brain Tumor Auto-Segmentation for MRI using 3D U-Net

..........MRI Segmentation Example..............

## Overview
This project implements a deep learning model based on the 3D U-Net architecture for automatic segmentation of brain tumors in MRI scans. The model identifies and classifies three distinct tumor sub-regions:
* **Edemas**
* **Non-Enhancing Tumors**
* **Enhancing Tumors**

⚕️ This work contributes to the growing field of AI-assisted radiology by supporting more accurate and efficient tumor detection, treatment planning  and monitoring of brain tumors.


##📄 Project Ispiration
This project is inspired by the paper:
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
by Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, and Olaf Ronneberger.

##🗂️ Dataset
This project utilizes data from the **Medical Segmentation Decathlon (MSD) MICCAI BraTS 2021 - Task 01 (Brain Tumor)** dataset.

**Modality:** The dataset consists of **multi-modal MRI** scans, including the following sequences:

* **FLAIR** (Fluid-Attenuated Inversion Recovery)
* **T1** (T1-weighted)
* **T1Gd** (T1-weighted with Gadolinium contrast enhancement)
* **T2** (T2-weighted)

**Labels:** The dataset provides **multi-class voxel-wise segmentation** labels for the following brain tumor sub-regions:

* **Non-Enhancing Tumor Core:** Label **1**
* **Edema:** Label **2**
* **Enhancing Tumor:** Label **4**

##🧠 Model Architecture
**Base Model:** 3D U-Net

**Key Techniques Employed:**

* **Patch-based training with sub-volume sampling:** This approach efficiently handles the large volumetric MRI data by training on smaller, randomly sampled sub-volumes (patches) of the full scans.
* **Volumetric convolution:** The model utilizes 3D convolutional layers to process the three-dimensional MRI volumes, capturing spatial information in all three dimensions.
* **Skip connections for spatial feature preservation:** Skip connections are implemented to directly link feature maps from the contracting path to the expanding path of the U-Net. This helps to preserve fine-grained spatial details crucial for accurate segmentation.


## 🧪 Evaluation Metrics

The performance of the brain tumor segmentation model was evaluated using the following key metrics:

* **Dice Coefficient:** A common metric for evaluating the overlap between the predicted segmentation and the ground truth. It ranges from 0 (no overlap) to 1 (perfect overlap).
* **Soft Dice Loss:** The loss function used during training, which is a differentiable form of the Dice coefficient, allowing for effective gradient-based optimization.
* **Class-wise Sensitivity and Specificity:** These metrics provide a detailed evaluation of the model's performance for each individual tumor sub-region (Edema, Non-Enhancing Tumor, Enhancing Tumor):
    * **Sensitivity (Recall):** Measures the model's ability to correctly identify all positive instances of a class.
    * **Specificity:** Measures the model's ability to correctly identify all negative instances of a class.

## 🚀 Training Setup

The model was trained with the following configuration:

* **GPU:** NVIDIA T4
* **Loss Function:** Soft Dice Loss
* **Patch Size:** Training was performed on randomly extracted sub-volumes (patches) from the full 3D MRI scans to manage memory constraints and improve learning.
* **Epochs:** The model was trained for over 100 epochs to allow for sufficient learning and convergence.
* **Batch Size:** 8
  

##📊 Results

The model achieved the following performance metrics during training and validation:

## 🏆Best Performance Metrics

| Metric            | Train   | Validation |
| :---------------- | :------ | :--------- |
| **Dice Loss** | 0.2667  | 0.2629     |
| **Dice Coefficient** | 0.7107  | 0.7273     |

## 🔍 Class-wise Performance

| Abnormality          | Sensitivity | Specificity |
| :------------------- | :---------- | :---------- |
| Edema                | 0.8272      | 0.9767      |
| Non-Enhancing Tumor  | 0.6384      | 0.9985      |
| Enhancing Tumor      | 0.9813      | 0.9872      |

### TensorBoard Visualizations

The following visualizations from TensorBoard illustrate the training progress of the Dice Coefficient and Loss for both the training and validation datasets, as well as the evaluation Dice Coefficient and Loss over iterations:

**Epoch Dice Coefficient:**

![Epoch Dice Coefficient](link_to_your_image1.png)

**Epoch Loss:**

![Epoch Loss](link_to_your_image2.png)

**Evaluation Dice Coefficient vs. Iterations:**

![Evaluation Dice Coefficient vs. Iterations](link_to_your_image3.png)

**Evaluation Loss vs. Iterations:**

![Evaluation Loss vs. Iterations](link_to_your_image4.png)

*Note: Please replace the `link_to_your_imageX.png` placeholders with the actual links or paths to your uploaded images within your GitHub repository.*



## Training Details

* **Training Strategy:** Patch-based training with sub-volume sampling.
* **Loss Function:** Soft Dice Loss.
* **Optimizer:** *(Specify your optimizer, e.g., Adam, SGD)*
* **Learning Rate:** *(Specify your initial learning rate and any scheduling used)*
* **Epochs:** *(Specify the number of training epochs)*
* **Batch Size:** *(Specify the batch size used during training)*
* **Hardware:** NVIDIA T4 GPU.

## Getting Started

**(Provide clear and concise instructions on how to set up and run your project. This might include:**

1.  **Prerequisites:**
    * Python 3.10
    * Tensorflow 2.18
    * GPU
    * Other necessary libraries NiBabel, NumPy, OpenCv, Matplotlib, Tensorboard
 * installation
    ```bash
    # Clone repository
    git clone https://github.com/matician255/BRAIN-TUMOUR-AUTOSEGMENTATION-FOR-MRI-SCANS.git
    cd brain-tumor-segmentation
    ```

2.  **Dataset Setup:**
  * Get the data at [Medical Segmentation Decathlon](https://decathlon-10.grand-challenge.org/)  

4.  **Running the Code:** Provide commands to train the model, evaluate it, or run inference on new data.

    ```bash
    # Sample Inference Script
    from tensorflow.keras.models import load_model
    
    # Define custom objects
    custom_objects = {
        "soft_dice_loss": soft_dice_loss,
        "dice_coefficient": dice_coefficient
    }
    
    # Load the model with custom objects
    loaded_model = load_model("/kaggle/input/saved-model/best_MRI_model.keras", custom_objects=custom_objects)
    case_id = 'BRATS_001'
    img_1, label_1 = load_case(case_id)
    prediction = predict_and_viz(img_1, label_1, loaded_model,threshold, loc=(100, 100, 50))
     # get the sensitivity and specificity for the whole scan
    whole_scan_df = get_sens_spec_df(whole_scan_pred, whole_scan_label)
  
    ```


## 🧠 Why This Matters

Accurate brain tumor segmentation from MRI scans is a crucial step in the clinical workflow for diagnosis, treatment planning, and monitoring of patients with brain tumors. However, manual segmentation is a time-consuming and labor-intensive task, and can be subject to inter-rater variability.

Automating this process using artificial intelligence offers significant advantages:

* **Reduced diagnostic workload for radiologists:** AI-powered segmentation can alleviate the burden on radiologists, allowing them to focus on more complex cases and improve overall efficiency.
* **Improved segmentation consistency:** Automated methods provide consistent and reproducible segmentations, reducing variability that can occur with manual annotation.
* **Speed up treatment planning in neuro-oncology:** Faster and more accurate segmentation can accelerate the treatment planning process, leading to quicker interventions and potentially better patient outcomes.


## 📌 Future Work

Building upon the current results, future work will focus on several key areas to further enhance the capabilities and clinical applicability of the brain tumor segmentation model:

* **Integrate clinical metadata for multi-modal prediction:** Incorporating patient-specific clinical information (e.g., age, tumor grade) alongside the multi-modal MRI data could potentially improve prediction accuracy and provide more context-aware segmentations.
* **Explore transfer learning with pre-trained medical models:** Investigating the use of transfer learning by leveraging weights from models pre-trained on large-scale medical imaging datasets could lead to faster convergence and improved generalization, especially with limited brain tumor data.
* **Deploy a web-based visualization tool for segmented MRIs:** Developing an interactive web-based tool would allow clinicians and researchers to easily visualize the segmented MRI volumes, facilitating interpretation and analysis of the results.


## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

## Acknowledgements

* The authors of the **"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"** paper for their foundational work.
* The organizers of the **DECATHLON 10** challenge for providing the valuable dataset.
* The developers of **PyTorch** and other open-source libraries used in this project.

## Contact

Emily Godfrey 

mathematiciangodfrey@outlook.com

https://www.linkedin.com/in/emilyemily255/

https://github.com/matician255
