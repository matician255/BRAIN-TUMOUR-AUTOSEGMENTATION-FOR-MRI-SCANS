import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Before any other imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info messages

import tensorflow as tf
from tensorflow.keras.layers import ( # type: ignore
    Input, Conv3D, MaxPool3D, UpSampling3D, 
    Conv3DTranspose, Concatenate, Activation,
    BatchNormalization
)
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from keras import backend as K
from garage import create_h5_patches,VolumeDataGenerator, unet_model_3d, soft_dice_loss, dice_coefficient
import h5py
from sklearn.model_selection import train_test_split


model = unet_model_3d(loss_function=soft_dice_loss, metrics=[dice_coefficient])


#process raw data into HDF5 into patches
output_path = r'D:\courses\AI FOR MEDICINE\MEDICAL DIAGNOSIS\tumour_project\BrainTumour_dataset\preprocessed_patches'
raw_dir = r'D:\courses\AI FOR MEDICINE\MEDICAL DIAGNOSIS\tumour_project\BrainTumour_dataset\imagesTr'

# Create directory if needed
os.makedirs(output_path, exist_ok=True)
output_path = os.path.join(output_path, 'train_patches.h5')


h5_path = r'D:\courses\AI FOR MEDICINE\MEDICAL DIAGNOSIS\tumour_project\BrainTumour_dataset\preprocessed_patches\train_patches.h5'
# Load data from H5 file
with h5py.File('your_data.h5', 'r') as f:
    X = f['X'][:]  # Features
    y = f['y'][:]   # Labels

# Split into train and validation sets (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,  # For reproducibility
    shuffle=True      # Shuffle data before splitting
)

print(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}")
if __name__ == "__main__":
    model.summary()
    # 1. Process raw data into HDF5 patches
    create_h5_patches(
        raw_dir=raw_dir,
        output_path=output_path,
        num_patches_per_scan=10,
        orig_shape=(240, 240, 155),
        patch_shape=(160, 160, 16)
    )

    # Add validation in create_h5_patches:
    print(f"Processing {len(os.listdir(raw_dir))} cases")
    print(f"Sample case: {os.listdir(raw_dir)[0]}")


# 2. Create generators
import os
train_h5_path='/content/drive/MyDrive/Colab Notebooks/dataset_h5/train_split.h5'
val_h5_path='/content/drive/MyDrive/Colab Notebooks/dataset_h5/val_split.h5'


# 2. Create generators
train_gen = VolumeDataGenerator(
    train_h5_path,
    batch_size=8,
    shuffle=True,
    augment=True
)

val_gen = VolumeDataGenerator(
    val_h5_path,
    batch_size=8,
    shuffle=False,
    augment=False
)

steps_per_epoch = len(train_gen)
n_epochs=5
validation_steps = len(val_gen)

CKPT_DIR= '/content/drive/MyDrive/Colab Notebooks/model_weights'

callbacks = [
    # Stop training when validation loss plateaus
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    ),
    #Per epoch weights
    # tf.keras.callbacks.ModelCheckpoint(
     #   filepath=os.path.join(CKPT_DIR, 'ckpt_epoch_{epoch:02d}.weights.h5'),
     #   save_weights_only=True,
     #   save_freq='epoch',
     #   verbose=1
     #),
    # Save best model automatically
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CKPT_DIR, 'best_model.h5'),
        monitor='val_dice_coefficient',
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),

    # Adjust learning rate dynamically
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10,
        min_lr=1e-6
    ),

    # Optional: TensorBoard for 3D visualization
    tf.keras.callbacks.TensorBoard(log_dir='/content/drive/MyDrive/Colab Notebooks/model_Logs', histogram_freq=1)
]



last_epoch = 99
# 3. Build and train model
model.fit(
    x=train_gen,
    validation_data=val_gen,
    epochs=last_epoch + n_epochs,
    initial_epoch=last_epoch,
    verbose=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks
)


#Native Keras format
model.save('/content/drive/MyDrive/Colab Notebooks/model_weights/best_MRI_model.keras')

# model evaluation
test_patch_path = '/content/drive/MyDrive/Colab Notebooks/dataset_h5/test_patches.h5'
test_gen = VolumeDataGenerator(
    test_patch_path,
    batch_size=8,
    shuffle=False,
    augment=False
)

model.load_weights('/content/drive/MyDrive/Colab Notebooks/model_weights/best_MRI_model.keras')
val_loss, val_dice = model.evaluate(test_gen)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Dice Coefficient: {val_dice:.4f}")