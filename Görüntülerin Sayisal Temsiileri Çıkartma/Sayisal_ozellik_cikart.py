import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_otsu
from scipy.stats import kurtosis, skew
from skimage.measure import shannon_entropy
import time

# Load PNG image
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

# Basic Volumetric Features
def extract_volumetric_features(image):
    brain_volume = np.sum(image > 0)
    gray_matter_volume = np.sum((image > 50) & (image <= 150))
    white_matter_volume = np.sum(image > 150)
    csf_volume = np.sum(image <= 50)
    return brain_volume, gray_matter_volume, white_matter_volume, csf_volume

# Histogram Features
def extract_histogram_features(image):
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    max_intensity = np.max(image)
    return mean_intensity, std_intensity, max_intensity

# Haralick Features
def extract_haralick_features(image):
    glcm = graycomatrix(image, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    return contrast, homogeneity, energy, correlation, asm

# Local Binary Patterns
def extract_lbp_features(image, P=8, R=1):
    lbp = local_binary_pattern(image, P, R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize the histogram
    return hist

# Calculate brain area
def calculate_brain_area(image):
    thresh = threshold_otsu(image)
    binary = image > thresh
    area = np.sum(binary)
    return area

# Calculate Fractal Dimension
def fractal_dimension(Z, threshold=0.9):
    Z = (Z < threshold)
    p = min(Z.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)
    counts = [boxcount(Z, size) for size in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def boxcount(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                           np.arange(0, Z.shape[1], k), axis=1)
    return len(np.where((S > 0) & (S < k*k))[0])

# Placeholder functions for new features
def extract_live_neurons(image):
    # Placeholder logic for live neuron count
    total_neurons = np.sum(image > 100)  # Example logic
    neuron_density = total_neurons / image.size
    neuron_distribution = np.var(image[image > 100])
    return total_neurons, neuron_density, neuron_distribution

def extract_random_forest_mapping(image):
    # Placeholder logic for random forest mapping regions
    # Normally, you would use a pre-trained model for this
    occipital_lobe = np.mean(image[:50, :50])
    hippocampus = np.mean(image[50:100, :50])
    temporal_lobe = np.mean(image[100:150, :50])
    thalamus = np.mean(image[:50, 50:100])
    insular_lobe = np.mean(image[50:100, 50:100])
    frontal_lobe = np.mean(image[100:150, 50:100])
    return occipital_lobe, hippocampus, temporal_lobe, thalamus, insular_lobe, frontal_lobe

# Manual RLM calculation function
def calculate_rlm_features(image):
    levels = 256
    rlm_features = {
        'SRE': [],
        'LRE': [],
        'GLN': [],
        'RLN': [],
        'RP': [],
        'LGRE': [],
        'HGRE': []
    }
    
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    for angle in angles:
        run_length_matrix = np.zeros((levels, image.size), dtype=int)
        
        for i in range(image.shape[0]):
            run_length = 1
            for j in range(1, image.shape[1]):
                if image[i, j] == image[i, j-1]:
                    run_length += 1
                else:
                    run_length_matrix[image[i, j-1], run_length-1] += 1
                    run_length = 1
            run_length_matrix[image[i, -1], run_length-1] += 1
        
        rlm_features['SRE'].append(np.sum(run_length_matrix / (np.arange(1, run_length_matrix.shape[1] + 1) ** 2)))
        rlm_features['LRE'].append(np.sum(run_length_matrix * (np.arange(1, run_length_matrix.shape[1] + 1) ** 2)))
        rlm_features['GLN'].append(np.sum(np.sum(run_length_matrix, axis=1) ** 2))
        rlm_features['RLN'].append(np.sum(np.sum(run_length_matrix, axis=0) ** 2))
        rlm_features['RP'].append(np.sum(run_length_matrix) / (image.shape[0] * image.shape[1]))
        rlm_features['LGRE'].append(np.sum(run_length_matrix / (np.arange(1, levels + 1) ** 2)[:, None]))
        rlm_features['HGRE'].append(np.sum(run_length_matrix * (np.arange(1, levels + 1) ** 2)[:, None]))

    # Averaging over all angles
    averaged_rlm_features = {key: np.mean(value) for key, value in rlm_features.items()}
    
    return averaged_rlm_features


# Extract features from MRI image
def extract_features(image_path):
    start_time = time.time()
    
    image = load_image(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Resize image to speed up processing
    image = cv2.resize(image, (128, 128))

    # Volumetric features
    brain_volume, gray_matter_volume, white_matter_volume, csf_volume = extract_volumetric_features(image)
    
    # Histogram features
    mean_intensity, std_intensity, max_intensity = extract_histogram_features(image)
    
    # Haralick features
    contrast, homogeneity, energy, correlation, asm = extract_haralick_features(image)
    
    # LBP features
    lbp_hist = extract_lbp_features(image)
    
    # Brain area
    area = calculate_brain_area(image)
    
    # Fractal Dimension
    fractal_dim = fractal_dimension(image)
    
    # Entropy
    entropy = shannon_entropy(image)
    
    # Skewness and Kurtosis
    skewness = skew(image.flatten())
    kurt = kurtosis(image.flatten())
    
    # Live Neurons Features
    total_neurons, neuron_density, neuron_distribution = extract_live_neurons(image)
    
    # Random Forest Mapping Regions
    occipital_lobe, hippocampus, temporal_lobe, thalamus, insular_lobe, frontal_lobe = extract_random_forest_mapping(image)
    
    # RLM Features
    rlm_features = calculate_rlm_features(image)
    
    features = {
        'brain_volume': brain_volume,
        'gray_matter_volume': gray_matter_volume,
        'white_matter_volume': white_matter_volume,
        'csf_volume': csf_volume,
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'max_intensity': max_intensity,
        'contrast': contrast,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation,
        'asm': asm,
        'area': area,
        'fractal_dim': fractal_dim,
        'entropy': entropy,
        'skewness': skewness,
        'kurtosis': kurt,
        'total_neurons': total_neurons,
        'neuron_density': neuron_density,
        'neuron_distribution': neuron_distribution,
        'occipital_lobe': occipital_lobe,
        'hippocampus': hippocampus,
        'temporal_lobe': temporal_lobe,
        'thalamus': thalamus,
        'insular_lobe': insular_lobe,
        'frontal_lobe': frontal_lobe
    }
    
    # Adding LBP histogram features separately
    for i, val in enumerate(lbp_hist):
        features[f'lbp_{i}'] = val
    
    # Adding RLM features separately
    features.update(rlm_features)

    end_time = time.time()
    print(f"Extracted features from {image_path} in {end_time - start_time:.2f} seconds.")
    
    return features

# Directory where images are stored
base_dir = 'processed_kirpilmamis'
folders = ['label_0', 'label_1', 'label_2', 'label_3']

# Initialize lists to store data
data = []

# Iterate over each label folder
for label_folder in folders:
    label_folder_path = os.path.join(base_dir, label_folder)

    # Skip if it's not a directory
    if not os.path.isdir(label_folder_path):
        continue

    # Iterate over each image in the label folder
    for image_name in sorted(os.listdir(label_folder_path)):
        if image_name.endswith('.png'):
            image_path = os.path.join(label_folder_path, image_name)

            # Extract features from the image
            features = extract_features(image_path)
            if features is not None:
                # Append additional info
                features["name"] = image_name
                features["label"] = label_folder

                # Append to list
                data.append(features)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_file_path = 'vk4.csv'
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved with {len(df)} entries: {csv_file_path}")
