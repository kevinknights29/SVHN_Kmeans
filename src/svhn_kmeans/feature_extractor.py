"""
K-means based feature extraction for digit recognition.
Implements the feature extraction pipeline described in the paper.
Supports GPU acceleration and parallel processing.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.signal import convolve2d
from .utils import rgb_to_gray, extract_patches, normalize_patch
import pickle
from typing import Tuple, List
from functools import partial

# Try to import GPU acceleration libraries
try:
    import cupy as cp
    from cupyx.scipy.signal import convolve2d as gpu_convolve2d
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

# Try to import parallel processing
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


class KMeansFeatureExtractor:
    """
    Feature extractor using K-means clustering to learn filters.

    The system:
    1. Learns dictionary of K filters (centroids) from image patches
    2. Convolves these filters with input images
    3. Applies non-linearity: g(z) = max{0, |z| - α}
    4. Performs spatial average pooling in a 5x5 grid
    """

    def __init__(self, K: int = 500, patch_size: Tuple[int, int] = (8, 8),
                 pool_grid: Tuple[int, int] = (5, 5),
                 use_gpu: bool = True, n_jobs: int = -1):
        """
        Initialize feature extractor.

        Args:
            K: Number of dictionary elements (cluster centroids)
            patch_size: Size of patches to extract (height, width)
            pool_grid: Size of pooling grid (rows, cols)
            use_gpu: Use GPU acceleration if available (requires CuPy)
            n_jobs: Number of parallel jobs (-1 for all cores, requires joblib)
        """
        self.K = K
        self.patch_size = patch_size
        self.pool_grid = pool_grid
        self.dictionary = None  # Will be (K, patch_h * patch_w)
        self.alpha = None  # Learned threshold for activation
        self.kmeans = None

        # GPU acceleration
        self.use_gpu = use_gpu and HAS_CUPY
        if use_gpu and not HAS_CUPY:
            print("Warning: CuPy not available.")
            print("  For CUDA 12.x: pip install cupy-cuda12x")
            print("  For CUDA 11.x: pip install cupy-cuda11x")
            print("  Check CUDA version: nvidia-smi")
            print("Falling back to CPU processing.")

        # Parallel processing
        self.n_jobs = n_jobs if HAS_JOBLIB else 1
        if n_jobs != 1 and not HAS_JOBLIB:
            print("Warning: joblib not available. Install with: pip install joblib")
            print("Falling back to sequential processing.")

        # GPU memory management
        if self.use_gpu:
            self.dictionary_gpu = None
            print(f"GPU acceleration enabled (CuPy detected)")

    def learn_dictionary(self, images: list, n_samples_per_image: int = 1000,
                        n_iterations: int = 100, batch_size: int = 10000,
                        verbose: bool = True):
        """
        Learn dictionary of filters using K-means clustering.

        Args:
            images: List of training images (can be grayscale or RGB)
            n_samples_per_image: Number of patches to sample from each image
            n_iterations: Maximum number of K-means iterations
            batch_size: Batch size for mini-batch K-means
            verbose: Print progress
        """
        if verbose:
            print(f"Extracting patches from {len(images)} images...")

        all_patches = []

        # Extract patches from images
        for idx, img in enumerate(images):
            if verbose and idx % 500 == 0:
                print(f"Processing image {idx}/{len(images)}")

            # Convert to grayscale
            gray = rgb_to_gray(img)

            # Extract patches
            patches = extract_patches(gray, self.patch_size, stride=1, normalize=True)

            # Sample random patches if we have too many
            if len(patches) > n_samples_per_image:
                indices = np.random.choice(len(patches), n_samples_per_image, replace=False)
                patches = patches[indices]

            all_patches.append(patches)

        all_patches = np.vstack(all_patches)

        if verbose:
            print(f"Total patches: {len(all_patches)}")
            print(f"Running K-means with K={self.K}...")

        # Run mini-batch K-means
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.K,
            max_iter=n_iterations,
            batch_size=batch_size,
            verbose=verbose,
            random_state=42
        )

        self.kmeans.fit(all_patches)

        # Store dictionary (centroids)
        self.dictionary = self.kmeans.cluster_centers_.copy()

        # Normalize each filter to unit norm
        for i in range(self.K):
            norm = np.linalg.norm(self.dictionary[i])
            if norm > 1e-5:
                self.dictionary[i] /= norm

        if verbose:
            print("Learning threshold α...")

        # Learn threshold α based on mean activation
        # Sample some patches and compute activations
        sample_size = min(50000, len(all_patches))
        sample_indices = np.random.choice(len(all_patches), sample_size, replace=False)
        sample_patches = all_patches[sample_indices]

        # Compute activations
        activations = np.abs(self.dictionary @ sample_patches.T)
        self.alpha = np.mean(activations)

        if verbose:
            print(f"Learned α = {self.alpha:.4f}")
            print("Dictionary learning complete!")

        # Upload dictionary to GPU if using GPU
        if self.use_gpu:
            self.dictionary_gpu = cp.asarray(self.dictionary)
            if verbose:
                print("Dictionary uploaded to GPU")

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from a 32x32 image using the learned dictionary.
        Uses GPU acceleration if available.

        Args:
            image: Input image (32x32, can be RGB or grayscale)

        Returns:
            Feature vector of shape (K * pool_grid[0] * pool_grid[1],)
        """
        if self.dictionary is None:
            raise ValueError("Dictionary not learned yet. Call learn_dictionary() first.")

        if self.use_gpu:
            return self._extract_features_gpu(image)
        else:
            return self._extract_features_cpu(image)

    def _extract_features_cpu(self, image: np.ndarray) -> np.ndarray:
        """CPU version of feature extraction."""
        # Convert to grayscale
        gray = rgb_to_gray(image)

        # Ensure image is 32x32
        if gray.shape != (32, 32):
            from .utils import resize_to_32x32
            gray = resize_to_32x32(gray)

        # Convolve each filter with the image
        patch_h, patch_w = self.patch_size
        conv_h = 32 - patch_h + 1  # 32 - 8 + 1 = 25
        conv_w = 32 - patch_w + 1  # 32 - 8 + 1 = 25

        activations = np.zeros((self.K, conv_h, conv_w))

        for k in range(self.K):
            # Reshape filter to 2D
            filter_2d = self.dictionary[k].reshape(self.patch_size)

            # Convolve filter with image
            conv_result = convolve2d(gray, filter_2d, mode='valid')

            # Apply non-linearity: g(z) = max{0, |z| - α}
            activations[k] = np.maximum(0, np.abs(conv_result) - self.alpha)

        # Spatial average pooling in pool_grid
        pooled_features = self._spatial_pooling(activations)

        return pooled_features

    def _extract_features_gpu(self, image: np.ndarray) -> np.ndarray:
        """GPU-accelerated version of feature extraction using CuPy."""
        # Convert to grayscale
        gray = rgb_to_gray(image)

        # Ensure image is 32x32
        if gray.shape != (32, 32):
            from .utils import resize_to_32x32
            gray = resize_to_32x32(gray)

        # Transfer to GPU
        gray_gpu = cp.asarray(gray, dtype=cp.float32)

        # Convolve each filter with the image
        patch_h, patch_w = self.patch_size
        conv_h = 32 - patch_h + 1
        conv_w = 32 - patch_w + 1

        activations_gpu = cp.zeros((self.K, conv_h, conv_w), dtype=cp.float32)

        for k in range(self.K):
            # Reshape filter to 2D
            filter_2d_gpu = self.dictionary_gpu[k].reshape(self.patch_size)

            # Convolve filter with image on GPU
            conv_result = gpu_convolve2d(gray_gpu, filter_2d_gpu, mode='valid')

            # Apply non-linearity: g(z) = max{0, |z| - α}
            activations_gpu[k] = cp.maximum(0, cp.abs(conv_result) - self.alpha)

        # Transfer back to CPU for pooling
        activations = cp.asnumpy(activations_gpu)

        # Spatial average pooling
        pooled_features = self._spatial_pooling(activations)

        return pooled_features

    def _spatial_pooling(self, activations: np.ndarray) -> np.ndarray:
        """
        Perform spatial average pooling over a grid.

        Args:
            activations: Activations of shape (K, H, W)

        Returns:
            Pooled features of shape (K * grid_h * grid_w,)
        """
        K, H, W = activations.shape
        grid_h, grid_w = self.pool_grid

        # Calculate pool region sizes
        pool_h = H // grid_h
        pool_w = W // grid_w

        pooled_features = []

        for k in range(K):
            for i in range(grid_h):
                for j in range(grid_w):
                    h_start = i * pool_h
                    h_end = min((i + 1) * pool_h, H)
                    w_start = j * pool_w
                    w_end = min((j + 1) * pool_w, W)

                    # Average pooling
                    pool_val = np.mean(activations[k, h_start:h_end, w_start:w_end])
                    pooled_features.append(pool_val)

        return np.array(pooled_features)

    def extract_features_batch(self, images: list, batch_size: int = 100,
                               verbose: bool = False) -> np.ndarray:
        """
        Extract features from multiple images with parallel processing.

        Args:
            images: List of images (each 32x32)
            batch_size: Batch size for progress reporting
            verbose: Show progress

        Returns:
            Feature matrix of shape (N, feature_dim)
        """
        n_images = len(images)

        if verbose:
            print(f"Extracting features from {n_images} images...")
            if self.use_gpu:
                print("Using GPU acceleration")
            if self.n_jobs > 1 or self.n_jobs == -1:
                print(f"Using parallel processing with {self.n_jobs} jobs")

        # Use parallel processing if available
        if HAS_JOBLIB and (self.n_jobs > 1 or self.n_jobs == -1):
            # Process in batches for better progress reporting
            all_features = []
            n_batches = (n_images + batch_size - 1) // batch_size

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_images)
                batch_images = images[start_idx:end_idx]

                # Parallel processing of batch
                batch_features = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                    delayed(self.extract_features)(img) for img in batch_images
                )

                all_features.extend(batch_features)

                if verbose and (batch_idx % 10 == 0 or batch_idx == n_batches - 1):
                    print(f"  Processed {end_idx}/{n_images} images")

            return np.array(all_features)
        else:
            # Sequential processing
            features = []
            for idx, img in enumerate(images):
                feat = self.extract_features(img)
                features.append(feat)

                if verbose and (idx % batch_size == 0 or idx == n_images - 1):
                    print(f"  Processed {idx + 1}/{n_images} images")

            return np.array(features)

    def save(self, filepath: str):
        """Save the learned dictionary and parameters."""
        data = {
            'K': self.K,
            'patch_size': self.patch_size,
            'pool_grid': self.pool_grid,
            'dictionary': self.dictionary,
            'alpha': self.alpha
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Feature extractor saved to {filepath}")

    def load(self, filepath: str):
        """Load a previously learned dictionary and parameters."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.K = data['K']
        self.patch_size = data['patch_size']
        self.pool_grid = data['pool_grid']
        self.dictionary = data['dictionary']
        self.alpha = data['alpha']
        print(f"Feature extractor loaded from {filepath}")

    def get_feature_dim(self) -> int:
        """Get the dimensionality of the feature vector."""
        return self.K * self.pool_grid[0] * self.pool_grid[1]
