
import tensorflow as tf
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input, Conv3D, MaxPool3D, UpSampling3D, 
    Conv3DTranspose, Concatenate, Activation,
    BatchNormalization)
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from keras import backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
from IPython.display import Image
import os 



# Function to visualize the MRI image and its corresponding label

def get_labeled_image(image, label, is_categorical=False): 
    """
    Update and return a labeled image for visualization.
    
    - If `is_categorical` is False, the function converts a label matrix (with integer labels) 
      into a one-hot encoded (categorical) format with 4 classes. (For example, for background, edema, 
      non-enhancing tumor, enhancing tumor.)
      
    - The input MRI image (a 4D array) is then normalized from the first channel.
    
    - A blank canvas (labeled_image) is created with the same size as a selected portion of the label.
      This canvas will eventually hold the background image (multiplied by a mask) and then overlay 
      colored labels representing the tumor regions.
      
    - The background (the healthy part of the image) is taken from the normalized image and assigned 
      to each of the three channels (red, green, and blue) of labeled_image.
      
    - Finally, the function overlays the label information (the non-background parts) in a bright color
      by multiplying those one-hot label channels by 255 and adding it to the image.
    """
    
    # Step 1: Convert label to categorical one-hot encoding if not already categorical.
    if not is_categorical:
        label = to_categorical(label, num_classes=4).astype(np.uint8)
    
    # Step 2: Normalize the input image.
    # We take only the first channel of the image and normalize its pixel values to be between 0 and 255.
    # cv2.normalize scales the values based on the minimum and maximum found in the image.
    # The normalized image is then converted to an unsigned 8-bit integer type.
    image = cv2.normalize(image[:, :, :, 0], None, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    
    # Step 3: Create an empty (zero) array for labeled_image that has the same shape as the label's channels 1 to end.
    # Typically, if label is one-hot encoded with 4 channels, label[:, :, :, 1:] gives channels corresponding to 
    # edema, non-enhancing tumor, and enhancing tumor.
    labeled_image = np.zeros_like(label[:, :, :, 1:])
    
    # Step 4: Remove the tumor part from the image (i.e., isolate the background).
    # We multiply the normalized image by the first channel of the label (label[:, :, :, 0]), which likely represents 
    # the background mask. The result is stored in all three channels of the labeled_image so that we have a grayscale 
    # background.
    labeled_image[:, :, :, 0] = image * (label[:, :, :, 0])
    labeled_image[:, :, :, 1] = image * (label[:, :, :, 0])
    labeled_image[:, :, :, 2] = image * (label[:, :, :, 0])
    
# Step 5: Overlay (color) the tumor labels.
    # For the tumor parts (channels 1 to end in the one-hot label), multiply by 255 so that they become bright (white)
    # and add them to the labeled_image. This creates an overlay that shows the tumor regions in a striking color.
    labeled_image += label[:, :, :, 1:] * 255
    
    return labeled_image






# visualize the labeled case

def visualize_case_grid(image, label=None, slice_idx=77):
    data_all = []

    data_all.append(image)

    fig, ax = plt.subplots(3, 6, figsize=[16, 9])

    # coronal plane
    coronal = np.transpose(data_all, [1, 3, 2, 4, 0])
    coronal = np.rot90(coronal, 1)

    # transversal plane
    transversal = np.transpose(data_all, [2, 1, 3, 4, 0])
    transversal = np.rot90(transversal, 2)

    # sagittal plane
    sagittal = np.transpose(data_all, [2, 3, 1, 4, 0])
    sagittal = np.rot90(sagittal, 1)

    for i in range(6):
        n = np.random.randint(coronal.shape[2])
        ax[0][i].imshow(np.squeeze(coronal[:, :, n, :]))
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])
        if i == 0:
            ax[0][i].set_ylabel('Coronal', fontsize=15)

    for i in range(6):
        n = np.random.randint(transversal.shape[2])
        ax[1][i].imshow(np.squeeze(transversal[:, :, n, :]))
        ax[1][i].set_xticks([])
        ax[1][i].set_yticks([])
        if i == 0:
            ax[1][i].set_ylabel('Transversal', fontsize=15)

    for i in range(6):
        n = np.random.randint(sagittal.shape[2])
        ax[2][i].imshow(np.squeeze(sagittal[:, :, n, :]))
        ax[2][i].set_xticks([])
        ax[2][i].set_yticks([])
        if i == 0:
            ax[2][i].set_ylabel('Sagittal', fontsize=15)

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()


# gif visualization
def visualize_data_gif(data_):
    os.makedirs("tmp", exist_ok=True) # Ensure the directory exists
    gif_path = os.path.join("tmp", "slice.gif") 
    images = []
    for i in range(data_.shape[0]):
        x = data_[min(i, data_.shape[0] - 1), :, :]
        y = data_[:, min(i, data_.shape[1] - 1), :]
        z = data_[:, :, min(i, data_.shape[2] - 1)]
        img = np.concatenate((x, y, z), axis=1)
        images.append(img)
    imageio.mimsave(gif_path, images, duration=0.01)
    return Image(filename=gif_path, format='png')

# visualize the patches
def visualize_patch(X, y):
    fig, ax = plt.subplots(1, 2, figsize=[10, 5], squeeze=False)

    ax[0][0].imshow(X[:, :, 0], cmap='Greys_r')
    ax[0][0].set_yticks([])
    ax[0][0].set_xticks([])
    ax[0][1].imshow(y[:, :, 0], cmap='Greys_r')
    ax[0][1].set_xticks([])
    ax[0][1].set_yticks([])

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    
 # implementing dice coefficient for single or multiple classes 
def dice_coefficient(y_true, y_pred, axis=(0, 1, 2), 
                                  epsilon=0.00001):
    """
    Compute dice coefficient for single class.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for single class.
                                    shape: (x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of predictions for single class.
                                    shape: (x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator of dice coefficient.
                      Hint: pass this as the 'axis' argument to the K.sum function.
        epsilon (float): small constant added to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_coefficient (float): computed value of dice coefficient.     
    """
    
    dice_numerator = (2 * K.sum(y_true * y_pred, axis = axis) ) + epsilon
    dice_denominator = K.sum(y_true, axis = axis) + K.sum(y_pred, axis = axis) + epsilon
    # incase of single class
    # dice_coefficient = dice_numerator / dice_denominator
    # incase of muiltiple classes
    dice_coefficient = K.mean(dice_coefficient, axis = 0)

    return dice_coefficient

"""
Since the dice coefficient expects outputs to be 0/1 but our model outputs 
probabilities so that's where soft dice formula comes in.
"""

def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3), 
                   epsilon=0.00001):
    """
    Compute mean soft dice loss over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of soft predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator in formula for dice loss.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant added to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_loss (float): computed value of dice loss.     
    """

    dice_numerator = (2 * K.sum(y_pred * y_true, axis = axis)) + epsilon
    dice_denominator = K.sum(y_pred * y_pred, axis = axis) + K.sum(y_true * y_true, axis= axis) + epsilon
    dice_loss = 1 - K.mean(dice_numerator/dice_denominator)

    return dice_loss


# creating and training the model

# 3D unet model

def create_conv_block(input_layer, n_filters, batch_normalization=False):
    """Create a convolution block with two 3x3x3 conv layers."""
    x = Conv3D(n_filters, (3, 3, 3), padding='same')(input_layer)
    if batch_normalization:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv3D(n_filters, (3, 3, 3), padding='same')(x)
    if batch_normalization:
        x = BatchNormalization()(x)
    return Activation('relu')(x)

def get_up_conv(n_filters, pool_size, deconvolution):
    """Return either transposed conv or upsampling layer"""
    if deconvolution:
        return Conv3DTranspose(n_filters, pool_size, strides=pool_size, padding='valid')
    else:
        return UpSampling3D(size=pool_size)

def unet_model_3d(loss_function, input_shape=(160, 160, 16, 4),  # Channels-last format
                 n_labels=3,
                 pool_size=(2, 2, 2),
                 depth=4,
                 n_base_filters=32,
                 batch_normalization=False,
                 deconvolution=False,
                 activation='sigmoid',
                 metrics=[],
                 initial_lr=1e-5):
    """
    3D U-Net implementation with modern TensorFlow conventions
    
    Args:
        input_shape: 4D tuple (x, y, z, channels) in channels-last format
        n_labels: Number of output labels
        pool_size: Pooling/upsampling factor
        depth: Network depth
        n_base_filters: Base filters multiplier
        batch_normalization: Use BN after convs
        deconvolution: Use transposed convs instead of upsampling
        activation: Final activation function
        initial_lr: Initial learning rate
    
    Returns:
        Compiled TensorFlow model
    """
    
    inputs = Input(input_shape)
    current_layer = inputs
    skips = []

    # Encoder Path
    for d in range(depth):
        # Double filters at each level
        filters = n_base_filters * (2 ** d)
        
        # Conv block
        conv = create_conv_block(current_layer, filters, batch_normalization)
        
        # Store skip connection
        skips.append(conv)
        
        # Add pooling except last level
        if d < depth - 1:
            current_layer = MaxPool3D(pool_size)(conv)

    # Decoder Path
    for d in reversed(range(depth-1)):
        # Get corresponding skip connection
        skip = skips[d]
        
        # Upsampling
        up_conv = get_up_conv(
            n_filters=skip.shape[-1],  # Match skip connection filters
            pool_size=pool_size,
            deconvolution=deconvolution
        )(current_layer)
        
        # Concatenate with skip connection
        merged = Concatenate(axis=-1)([up_conv, skip])
        
        # Conv block
        current_layer = create_conv_block(
            merged, 
            skip.shape[-1],  # Maintain filter count
            batch_normalization
        )

    # Final convolution
    output = Conv3D(n_labels, (1, 1, 1))(current_layer)
    output = Activation(activation)(output)

    model = Model(inputs=inputs, outputs=output)
   
    
    model.compile(
        optimizer=Adam(learning_rate=initial_lr),
        loss=loss_function,  # Default - can be overridden
        metrics=metrics
    )
    
    model.load_weights(r'D:\courses\AI FOR MEDICINE\MEDICAL DIAGNOSIS\tumour_project\model_weights\ckpt_epoch_104.weights.h5')
    return model

# creating patches from the dataset
import h5py
import os
import numpy as np
from tqdm import tqdm
import data_viz, data_prep
from data_prep import get_sub_volume, standardize

def create_h5_patches(raw_dir, output_path, num_patches_per_scan=10,
                      orig_shape=(240, 240, 155), patch_shape=(160, 160, 16),
                      num_classes=4, background_threshold=0.95):
    """
    Process entire dataset into patch-based HDF5 format
    
    Args:
        raw_dir: Directory containing {image.nii, label.nii} pairs
        output_path: Path for output HDF5 file
        num_patches_per_scan: Number of patches to extract per scan
        orig_shape: Original image dimensions (x, y, z)
        patch_shape: Desired patch dimensions (x, y, z)
    """
    with h5py.File(output_path, 'w') as hf:
        # Create resizable datasets
        hf.create_dataset('X', shape=(0, 4, *patch_shape), 
                         maxshape=(None, 4, *patch_shape),
                         chunks=(1, 4, *patch_shape),
                         compression='gzip')
        
        hf.create_dataset('y', shape=(0, num_classes-1, *patch_shape),
                         maxshape=(None, num_classes-1, *patch_shape),
                         chunks=(1, num_classes-1, *patch_shape),
                         compression='gzip')

        # Process each scan
        # Filter out hidden files and non-NIfTI files
        valid_files = [f for f in os.listdir(raw_dir) 
                      if f.endswith('.nii.gz') and not f.startswith('._')]
        
        for scan_id in tqdm(valid_files):
            # Remove existing extension before adding .nii.gz
            case_id = os.path.splitext(os.path.splitext(scan_id)[0])[0]

            # Load raw data (modify for your file format)
            image, label = data_viz.load_case(case_id)
            # img_path = os.path.join(raw_dir, scan_id, 'image.nii')
            # label_path = os.path.join(raw_dir, scan_id, 'label.nii')
            
            # image = load_nifti(img_path)  # Implement your loader
            # label = load_nifti(label_path)
            
            # Preprocess
            # image = preprocess_volume(image)  # Implement normalization
            
            # Extract multiple patches per scan
            for _ in range(num_patches_per_scan):
                patch = get_sub_volume(
                    image, label,
                    orig_x=orig_shape[0], orig_y=orig_shape[1], orig_z=orig_shape[2],
                    output_x=patch_shape[0], output_y=patch_shape[1], output_z=patch_shape[2],
                    num_classes=num_classes,
                    background_threshold=background_threshold
                )
                if patch is None:
                    print(f"⚠️  Warning: no patch found for {scan_id}, retrying…")
                    continue
                X_patch, y_patch = patch
                
                # Standardize the patches
                X_patch = standardize(X_patch)
                
                # Append to HDF5
                hf['X'].resize((hf['X'].shape[0] + 1), axis=0)
                hf['X'][-1] = X_patch
                
                hf['y'].resize((hf['y'].shape[0] + 1), axis=0)
                hf['y'][-1] = y_patch


# Data generator
import numpy as np
class VolumeDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, h5_path, batch_size=8, shuffle=True, augment=False):
        """
        Args:
            h5_path: Path to HDF5 file containing all patches
            batch_size: Number of patches per batch
            shuffle: Whether to shuffle patch order
            augment: Whether to apply data augmentation
        """
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        with h5py.File(h5_path, 'r') as f:
            self.num_patches = f['X'].shape[0]

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(self.num_patches)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        # denotes the number of batches per epoch
        return int(np.ceil(self.num_patches / self.batch_size))

    def __getitem__(self, index):
        # generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:
                                min((index+1) * self.batch_size, self.num_patches)]

        with h5py.File(self.h5_path, 'r') as f:
          # Read samples one by one (avoiding h5py's index order restriction)
            X_batch = np.stack([f['X'][i] for i in indexes])
            y_batch = np.stack([f['y'][i] for i in indexes])

        if self.augment:
            X_batch, y_batch = self.apply_augmentation(X_batch, y_batch)

         # In __getitem__ after loading X_batch/y_batch:
        X_batch = np.moveaxis(X_batch, 1, -1)  # From (B, C, H, W, D) → (B, H, W, D, C)
        y_batch = np.moveaxis(y_batch, 1, -1)

        return X_batch, y_batch



    def apply_augmentation(self, X_batch, y_batch):
        """3D augmentation pipeline"""
        # Random flips
        if np.random.rand() > 0.5:
            axis = np.random.choice([1, 2, 3])  # Spatial axes only
            X_batch = np.flip(X_batch, axis)
            y_batch = np.flip(y_batch, axis)

        # Random rotations (0°, 90°, 180°, 270°)
        k = np.random.randint(0, 4)
        X_batch = np.rot90(X_batch, k=k, axes=(2, 3))  # Rotate Y-Z plane
        y_batch = np.rot90(y_batch, k=k, axes=(2, 3))

        # Random intensity scaling
        scale_factor = np.random.uniform(0.9, 1.1)
        X_batch = X_batch * scale_factor

        return X_batch, y_batch


# splittng a train patch into train and validation patch 
import h5py
from sklearn.model_selection import train_test_split

def split_and_save_h5(original_h5_path, train_h5_path, val_h5_path, test_size=0.2):
    with h5py.File(original_h5_path, 'r') as f_orig:
        num_samples = f_orig['X'].shape[0]
        indices = np.arange(num_samples)
        train_idx, val_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=42,
            shuffle=True
        )

        # Get dataset properties
        sample_shape_X = f_orig['X'].shape[1:]
        sample_shape_y = f_orig['y'].shape[1:]
        dtype_X = f_orig['X'].dtype
        dtype_y = f_orig['y'].dtype

        # Create train.h5
        with h5py.File(train_h5_path, 'w') as f_train:
            f_train.create_dataset(
                'X',
                shape=(len(train_idx), *sample_shape_X),
                dtype=dtype_X,
                chunks=(1, *sample_shape_X),  # For memory efficiency
                compression='gzip'
            )
            f_train.create_dataset(
                'y',
                shape=(len(train_idx), *sample_shape_y),
                dtype=dtype_y,
                chunks=(1, *sample_shape_y),
                compression='gzip'
            )

            # Populate train data incrementally
            for i, idx in enumerate(train_idx):
                f_train['X'][i] = f_orig['X'][idx]
                f_train['y'][i] = f_orig['y'][idx]

        # Create val.h5
        with h5py.File(val_h5_path, 'w') as f_val:
            f_val.create_dataset(
                'X',
                shape=(len(val_idx), *sample_shape_X),
                dtype=dtype_X,
                chunks=(1, *sample_shape_X),
                compression='gzip'
            )
            f_val.create_dataset(
                'y',
                shape=(len(val_idx), *sample_shape_y),
                dtype=dtype_y,
                chunks=(1, *sample_shape_y),
                compression='gzip'
            )

            # Populate validation data incrementally
            for i, idx in enumerate(val_idx):
                f_val['X'][i] = f_orig['X'][idx]
                f_val['y'][i] = f_orig['y'][idx]



if __name__ == "__main__":
    # Usage
    split_and_save_h5(
        original_h5_path='/content/drive/MyDrive/Colab Notebooks/dataset_h5/train_patches.h5',
        train_h5_path='/content/drive/MyDrive/Colab Notebooks/dataset_h5/train_split.h5',
        val_h5_path='/content/drive/MyDrive/Colab Notebooks/dataset_h5/val_split.h5'
    )


def sens_spec_compute(pred, label, class_num):
    """
    Compute sensitivity and specificity for a particular example
    for a given class.

    Args:
        pred (np.array): binary arrary of predictions, shape is
                         (num classes, height, width, depth).
        label (np.array): binary array of labels, shape is
                          (num classes, height, width, depth).
        class_num (int): number between 0 - (num_classes -1) which says
                         which prediction class to compute statistics
                         for.

    Returns:
        sensitivity (float): precision for given class_num.
        specificity(float): recall for given class_num """

    # extract the subarray for the specified class
    class_pred = pred[:, :, :, class_num]
    class_label = label[class_num]

    #true positive
    tp = np.sum((class_pred == 1)&(class_label == 1))

    #true negative
    tn = np.sum((class_pred == 0)&(class_label == 0))
    
    #false positive
    fp = np.sum((class_pred == 1)&(class_label == 0))

    #false negative 
    fn = np.sum((class_pred == 0)& (class_label == 1))


    #compute sensitivity and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity   


# Running on the entire scan 
def predict_and_viz(image, label, model, threshold, loc=(100, 100, 50)):
    image_labeled = get_labeled_image(image.copy(), label.copy())

    model_label = np.zeros([320, 320, 160, 3])

    for x in range(0, image.shape[0], 160):
        for y in range(0, image.shape[1], 160):
            for z in range(0, image.shape[2], 16):
                patch = np.zeros([160, 160, 16, 4])
                p = image[x: x + 160, y: y + 160, z:z + 16]
                print(p.shape) # (160, 160, 16, 4)
                patch[0:p.shape[0], 0:p.shape[1], 0:p.shape[2]] = p
                print(f"patch shape before expand: {patch.shape}") # (160, 160, 16, 4)
                pred = model.predict(np.expand_dims(patch, 0)) #patch = (1,160,160,16, 4)
                print(f"prediction shape: {pred.shape}") # (160, 160, 16, 3)
                
                model_label[x:x + p.shape[0],
                               y:y + p.shape[1],
                               z: z + p.shape[2], :] += pred[0][:p.shape[0], :p.shape[1],
                                      :p.shape[2], :]

    model_label = model_label[0:240, 0:240, 0:155, :]
    model_label_reformatted = np.zeros((240, 240, 155, 4))

    model_label_reformatted = to_categorical(label, num_classes=4).astype(
        np.uint8)

    model_label_reformatted[:, :, :, 1:4] = model_label

    model_labeled_image = get_labeled_image(image, model_label_reformatted,
                                            is_categorical=True)

    fig, ax = plt.subplots(2, 3, figsize=[10, 7])

    # plane values
    x, y, z = loc

    ax[0][0].imshow(np.rot90(image_labeled[x, :, :, :]))
    ax[0][0].set_ylabel('Ground Truth', fontsize=15)
    ax[0][0].set_xlabel('Sagital', fontsize=15)

    ax[0][1].imshow(np.rot90(image_labeled[:, y, :, :]))
    ax[0][1].set_xlabel('Coronal', fontsize=15)

    ax[0][2].imshow(np.squeeze(image_labeled[:, :, z, :]))
    ax[0][2].set_xlabel('Transversal', fontsize=15)

    ax[1][0].imshow(np.rot90(model_labeled_image[x, :, :, :]))
    ax[1][0].set_ylabel('Prediction', fontsize=15)

    ax[1][1].imshow(np.rot90(model_labeled_image[:, y, :, :]))
    ax[1][2].imshow(model_labeled_image[:, :, z, :])

    fig.subplots_adjust(wspace=0, hspace=.12)

    for i in range(2):
        for j in range(3):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])

    return model_label_reformatted
