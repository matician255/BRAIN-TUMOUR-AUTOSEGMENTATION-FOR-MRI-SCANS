import h5py
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import data_prep, data_viz # custom library to visualize the images
from garage import unet_model_3d, soft_dice_loss, dice_coefficient, visualize_patch, sens_spec_compute


img_id = 'BRATS_003' # example case id

img, label = data_viz.load_case(img_id) # loading the image and label
img_patch, label_patch = data_prep.get_sub_volume(img, label)

# Load the model weights
from tensorflow.keras.models import load_model

 
# Define custom objects
custom_objects = {
    "soft_dice_loss": soft_dice_loss,
    "dice_coefficient": dice_coefficient
}

# Load the model with custom objects
loaded_model = load_model("/kaggle/input/saved-model/best_MRI_model.keras", custom_objects=custom_objects)

# Load the model weights
# model = unet_model_3d(loss_function=soft_dice_loss, metrics=[dice_coefficient])

# Adding a batch dimension to the image patch 
img_patch_batch_dm = np.expand_dims(img_patch, axis=0) # shape (1, 4, 160, 160, 16)
# Original input shape: (1, 4, 160, 160, 16)
img_patch_batch_dm = np.transpose(img_patch_batch_dm, (0, 2, 3, 4, 1))# New shape: (1, 160, 160, 16, 4)

# patch prediction 
patch_prediction = loaded_model.predict(img_patch_batch_dm)
patch_prediction_shape = patch_prediction.shape
print(patch_prediction_shape)

#conversion from probability to category
threshold = 0.9
patch_prediction[patch_prediction > threshold] = 1
patch_prediction[patch_prediction <= threshold] = 0

print(f"patch and ground truth")
visualize_patch(img_patch[0, :, :, :], label_patch[2])
print(f"patch and prediction")
visualize_patch(img_patch[0, :, :, :], patch_prediction[0, :, :, :, 2])


import pandas as pd 

# sensitivity and specificity for each class
def get_sens_spec_df(pred, label):
    patch_metrics = pd.DataFrame(
        columns = ['Edema', 
                   'Non-Enhancing Tumor', 
                   'Enhancing Tumor'], 
        index = ['Sensitivity',
                 'Specificity'])
    
    for i, class_name in enumerate(patch_metrics.columns):
        sens, spec = sens_spec_compute(pred, label, i)
        patch_metrics.loc['Sensitivity', class_name] = round(sens,4)
        patch_metrics.loc['Specificity', class_name] = round(spec,4)

    return patch_metrics
