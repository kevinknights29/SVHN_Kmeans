"""
End-to-end pipeline for SVHN digit recognition.
"""

import numpy as np
import os
from .feature_extractor import KMeansFeatureExtractor
from .classifier import DigitClassifier
from .recognizer import BeamSearchRecognizer
from .utils import visualize_predictions, draw_bounding_boxes
from typing import List, Dict, Tuple, Union, Optional


class SVHNRecognitionPipeline:
    """
    End-to-end pipeline for recognizing house numbers in images.

    This is a simplified version that assumes the input is already
    a cropped house number region. For a full system, you would add
    a detection stage before this.
    """

    def __init__(self, feature_extractor: KMeansFeatureExtractor = None,
                 classifier: DigitClassifier = None,
                 beam_width: int = 10):
        """
        Initialize pipeline.

        Args:
            feature_extractor: Trained feature extractor (or None to create new)
            classifier: Trained classifier (or None to create new)
            beam_width: Beam width for recognition
        """
        self.feature_extractor = feature_extractor or KMeansFeatureExtractor()
        self.classifier = classifier or DigitClassifier()
        self.recognizer = None
        self.is_trained = False

        self.beam_width = beam_width

    def train(self, train_images: Union[List[np.ndarray], np.ndarray],
              train_labels: Union[List[int], np.ndarray],
              n_samples_per_image: int = 1000, verbose: bool = True,
              checkpoint_dir: Optional[str] = None,
              cache_features: bool = True):
        """
        Train the full pipeline.

        Args:
            train_images: List or array of 32x32 training images
            train_labels: List or array of labels (1-10 format, where 10 = digit 0)
            n_samples_per_image: Number of patches per image for dictionary learning
            verbose: Print progress
            checkpoint_dir: Directory to save checkpoints (None to disable)
            cache_features: If True, cache extracted features to disk
        """
        if verbose:
            print("=" * 60)
            print("Training SVHN Recognition Pipeline")
            print("=" * 60)

        # Step 1: Learn K-means dictionary
        if verbose:
            print("\n[1/3] Learning K-means dictionary...")

        # Check for cached dictionary
        dict_cache_file = None
        if checkpoint_dir:
            dict_cache_file = os.path.join(checkpoint_dir, 'kmeans_dictionary.pkl')
            if os.path.exists(dict_cache_file):
                if verbose:
                    print(f"✓ Loading K-means dictionary from cache: {dict_cache_file}")
                self.feature_extractor.load(dict_cache_file)
                if verbose:
                    print(f"✓ Loaded dictionary with K={self.feature_extractor.K} filters (skipped K-means training)")
            else:
                # Train and cache
                self.feature_extractor.learn_dictionary(
                    train_images,
                    n_samples_per_image=n_samples_per_image,
                    verbose=verbose
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                self.feature_extractor.save(dict_cache_file)
                if verbose:
                    print(f"✓ Cached K-means dictionary to {dict_cache_file}")
        else:
            # No caching
            self.feature_extractor.learn_dictionary(
                train_images,
                n_samples_per_image=n_samples_per_image,
                verbose=verbose
            )

        # Step 2: Extract features for all training images
        if verbose:
            print("\n[2/3] Extracting features from training images...")

        # Convert to list if numpy array
        if isinstance(train_images, np.ndarray):
            images_list = list(train_images)
        else:
            images_list = train_images

        # Check for cached features
        features_cache_file = None
        if cache_features and checkpoint_dir:
            features_cache_file = os.path.join(checkpoint_dir, 'train_features.npy')
            if os.path.exists(features_cache_file):
                if verbose:
                    print(f"Loading cached features from {features_cache_file}...")
                train_features = np.load(features_cache_file)
                if verbose:
                    print(f"✓ Loaded features: shape {train_features.shape}")
            else:
                # Extract and cache
                train_features = self.feature_extractor.extract_features_batch(
                    images_list,
                    batch_size=1000,
                    verbose=verbose
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                np.save(features_cache_file, train_features)
                if verbose:
                    print(f"✓ Cached features to {features_cache_file}")
        else:
            # Use batch processing with parallel/GPU acceleration
            train_features = self.feature_extractor.extract_features_batch(
                images_list,
                batch_size=1000,
                verbose=verbose
            )

        train_labels = np.array(train_labels)

        # Step 3: Train classifier
        if verbose:
            print(f"\n[3/3] Training L2 classifier...")

        self.classifier.train(
            train_features,
            train_labels,
            verbose=verbose,
            checkpoint_dir=checkpoint_dir,
            resume=True
        )

        # Create recognizer
        self.recognizer = BeamSearchRecognizer(
            self.feature_extractor,
            self.classifier,
            beam_width=self.beam_width
        )

        self.is_trained = True

        if verbose:
            print("\n" + "=" * 60)
            print("Training complete!")
            print("=" * 60)

    def evaluate(self, test_images: Union[List[np.ndarray], np.ndarray],
                test_labels: Union[List[int], np.ndarray],
                verbose: bool = True) -> Dict:
        """
        Evaluate pipeline on test data.

        Args:
            test_images: List or array of test images (32x32)
            test_labels: List or array of true labels
            verbose: Print results

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Pipeline not trained yet. Call train() first.")

        if verbose:
            print("Evaluating on test set...")

        # Convert to list if numpy array
        if isinstance(test_images, np.ndarray):
            images_list = list(test_images)
        else:
            images_list = test_images

        # Extract features
        test_features = self.feature_extractor.extract_features_batch(
            images_list,
            batch_size=1000,
            verbose=verbose
        )

        # Evaluate classifier
        results = self.classifier.evaluate(test_features, np.array(test_labels))

        if verbose:
            print(f"\nOverall Accuracy: {results['accuracy']:.4f}")
            print("\nPer-class accuracies:")
            for digit, acc in sorted(results['per_class_accuracy'].items()):
                print(f"  Digit {digit}: {acc:.4f}")

        return results

    def predict_single_digit(self, image: np.ndarray) -> int:
        """
        Predict a single digit (for 32x32 images).

        Args:
            image: Single digit image (32x32)

        Returns:
            Predicted digit (0-9)
        """
        if not self.is_trained:
            raise ValueError("Pipeline not trained yet.")

        features = self.feature_extractor.extract_features(image)
        prediction = self.classifier.predict(features.reshape(1, -1))[0]

        return int(prediction)

    def predict_multi_digit(self, image: np.ndarray,
                           return_visualization: bool = False) -> Tuple:
        """
        Predict multi-digit house number.

        Args:
            image: House number image (can be any size)
            return_visualization: If True, return annotated image

        Returns:
            If return_visualization is False: List of predicted digits
            If return_visualization is True: (digits, annotated_image, bboxes)
        """
        if not self.is_trained:
            raise ValueError("Pipeline not trained yet.")

        # Recognize digits
        results = self.recognizer.recognize(image, return_top_k=1)

        if not results:
            if return_visualization:
                return [], image, []
            return []

        best_result = results[0]
        digits = best_result['digits']
        bboxes = best_result['bboxes']

        if return_visualization:
            # Create visualization
            annotated = visualize_predictions(image, digits, bboxes)
            return digits, annotated, bboxes

        return digits

    def process_image(self, image: np.ndarray) -> Dict:
        """
        Process an image and return detailed results.

        Args:
            image: Input image (house number)

        Returns:
            Dictionary with results:
                - 'digits': Predicted digit sequence
                - 'confidence': Confidence score
                - 'bboxes': Bounding boxes
                - 'visualization': Annotated image
        """
        if not self.is_trained:
            raise ValueError("Pipeline not trained yet.")

        results = self.recognizer.recognize(image, return_top_k=1)

        if not results:
            return {
                'digits': [],
                'confidence': 0.0,
                'bboxes': [],
                'visualization': image
            }

        best_result = results[0]

        # Create visualization
        annotated = visualize_predictions(
            image,
            best_result['digits'],
            best_result['bboxes']
        )

        return {
            'digits': best_result['digits'],
            'confidence': best_result['score'],
            'bboxes': best_result['bboxes'],
            'visualization': annotated
        }

    def save(self, output_dir: str):
        """
        Save the trained pipeline.

        Args:
            output_dir: Directory to save model files
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Save components
        self.feature_extractor.save(os.path.join(output_dir, 'feature_extractor.pkl'))
        self.classifier.save(os.path.join(output_dir, 'classifier.pkl'))

        print(f"Pipeline saved to {output_dir}")

    def load(self, model_dir: str):
        """
        Load a trained pipeline.

        Args:
            model_dir: Directory containing saved model files
        """
        import os

        # Load components
        self.feature_extractor.load(os.path.join(model_dir, 'feature_extractor.pkl'))
        self.classifier.load(os.path.join(model_dir, 'classifier.pkl'))

        # Create recognizer
        self.recognizer = BeamSearchRecognizer(
            self.feature_extractor,
            self.classifier,
            beam_width=self.beam_width
        )

        self.is_trained = True

        print(f"Pipeline loaded from {model_dir}")
