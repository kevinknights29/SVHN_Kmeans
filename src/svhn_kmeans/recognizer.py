"""
Beam search recognizer for multi-digit house numbers.
"""

import numpy as np
from .segmentation import CharacterSegmenter, GeometryModel
from .feature_extractor import KMeansFeatureExtractor
from .classifier import DigitClassifier
from .utils import resize_to_32x32
from typing import List, Dict, Tuple


class BeamSearchRecognizer:
    """
    Recognizes multi-digit house numbers using beam search over segmentations.

    Combines:
    - Character segmentation with multiple hypotheses
    - Feature extraction for each candidate
    - Classification scores
    - Geometric constraints
    """

    def __init__(self, feature_extractor: KMeansFeatureExtractor,
                 classifier: DigitClassifier,
                 beam_width: int = 10,
                 classifier_weight: float = 0.7,
                 geometry_weight: float = 0.3,
                 max_digits: int = 6):
        """
        Initialize beam search recognizer.

        Args:
            feature_extractor: Trained feature extractor
            classifier: Trained digit classifier
            beam_width: Number of hypotheses to maintain
            classifier_weight: Weight for classifier score
            geometry_weight: Weight for geometry score
            max_digits: Maximum number of digits in a house number
        """
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.beam_width = beam_width
        self.classifier_weight = classifier_weight
        self.geometry_weight = geometry_weight
        self.max_digits = max_digits

        self.segmenter = CharacterSegmenter()
        self.geometry_model = GeometryModel()

    def recognize(self, image: np.ndarray, return_top_k: int = 5) -> List[Dict]:
        """
        Recognize digits in a house number image using beam search.

        Args:
            image: House number image (can be any size)
            return_top_k: Number of top hypotheses to return

        Returns:
            List of top-K hypotheses, each as dict with:
                - 'digits': List of predicted digits
                - 'score': Total score
                - 'bboxes': List of bounding boxes for each digit
        """
        # Find breakpoints for segmentation
        breakpoints = self.segmenter.find_breakpoints(image)

        # Initialize beam with empty path
        beam = [{
            'digits': [],
            'bboxes': [],
            'score': 0.0,
            'last_breakpoint_idx': 0
        }]

        final_paths = []

        # Beam search
        while beam:
            new_beam = []

            for path_state in beam:
                last_bp_idx = path_state['last_breakpoint_idx']

                # If we've covered the full width, this is a complete path
                if last_bp_idx >= len(breakpoints) - 1:
                    final_paths.append(path_state)
                    continue

                # Try extending path with each possible next character
                for next_bp_idx in range(last_bp_idx + 1, len(breakpoints)):
                    left = breakpoints[last_bp_idx]
                    right = breakpoints[next_bp_idx]
                    width = right - left

                    # Check if width is valid
                    if width < self.segmenter.min_char_width:
                        continue
                    if width > self.segmenter.max_char_width:
                        continue

                    # Extract character segment
                    segment = image[:, left:right]

                    # Resize to 32x32
                    segment_resized = resize_to_32x32(segment, maintain_aspect=True)

                    # Extract features
                    try:
                        features = self.feature_extractor.extract_features(segment_resized)
                    except Exception as e:
                        continue

                    # Classify - get scores for all 10 classes
                    class_scores = self.classifier.decision_function(features.reshape(1, -1))[0]

                    # Get top-K classes to explore
                    top_k_classes = np.argsort(class_scores)[-3:]  # Top 3 classes

                    for digit_class in top_k_classes:
                        # Compute scores
                        classifier_score = class_scores[digit_class]

                        bbox_info = {
                            'left': left,
                            'right': right,
                            'width': width,
                            'height': image.shape[0]
                        }
                        geo_score = self.geometry_model.score(bbox_info)

                        # Combined score
                        combined_score = (self.classifier_weight * classifier_score +
                                        self.geometry_weight * geo_score)

                        # Create new path
                        new_path = {
                            'digits': path_state['digits'] + [int(digit_class)],
                            'bboxes': path_state['bboxes'] + [bbox_info],
                            'score': path_state['score'] + combined_score,
                            'last_breakpoint_idx': next_bp_idx
                        }

                        # Check if we've reached the end
                        if next_bp_idx == len(breakpoints) - 1:
                            # Path is complete
                            if len(new_path['digits']) <= self.max_digits:
                                final_paths.append(new_path)
                        else:
                            # Continue exploring
                            if len(new_path['digits']) < self.max_digits:
                                new_beam.append(new_path)

            # Keep only top-N paths (beam pruning)
            new_beam.sort(key=lambda x: x['score'], reverse=True)
            beam = new_beam[:self.beam_width]

        # Return top-K complete paths
        final_paths.sort(key=lambda x: x['score'], reverse=True)

        # If no complete paths found, try greedy segmentation
        if not final_paths:
            final_paths = [self._greedy_recognition(image)]

        return final_paths[:return_top_k]

    def _greedy_recognition(self, image: np.ndarray) -> Dict:
        """
        Fallback greedy recognition when beam search fails.

        Args:
            image: House number image

        Returns:
            Recognition result
        """
        segments = self.segmenter.segment_greedy(image)

        digits = []
        bboxes = []
        total_score = 0.0

        for seg_info in segments:
            segment = seg_info['segment']
            segment_resized = resize_to_32x32(segment, maintain_aspect=True)

            try:
                features = self.feature_extractor.extract_features(segment_resized)
                prediction = self.classifier.predict(features.reshape(1, -1))[0]
                score = np.max(self.classifier.decision_function(features.reshape(1, -1))[0])

                digits.append(int(prediction))
                bboxes.append({
                    'left': seg_info['left'],
                    'right': seg_info['right'],
                    'width': seg_info['width'],
                    'height': image.shape[0]
                })
                total_score += score
            except Exception:
                continue

        return {
            'digits': digits,
            'bboxes': bboxes,
            'score': total_score
        }

    def recognize_simple(self, image: np.ndarray) -> List[int]:
        """
        Simple recognition that returns just the digit sequence.

        Args:
            image: House number image

        Returns:
            List of predicted digits
        """
        results = self.recognize(image, return_top_k=1)
        if results:
            return results[0]['digits']
        return []


class SequenceRecognizer:
    """
    Simpler recognizer for already-segmented digits.
    """

    def __init__(self, feature_extractor: KMeansFeatureExtractor,
                 classifier: DigitClassifier):
        """
        Initialize sequence recognizer.

        Args:
            feature_extractor: Trained feature extractor
            classifier: Trained digit classifier
        """
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def recognize_sequence(self, digit_images: List[np.ndarray]) -> List[int]:
        """
        Recognize a sequence of already-segmented digit images.

        Args:
            digit_images: List of digit images (can be any size)

        Returns:
            List of predicted digits
        """
        predictions = []

        for img in digit_images:
            # Resize to 32x32
            img_resized = resize_to_32x32(img, maintain_aspect=True)

            # Extract features
            features = self.feature_extractor.extract_features(img_resized)

            # Predict
            pred = self.classifier.predict(features.reshape(1, -1))[0]
            predictions.append(int(pred))

        return predictions

    def recognize_batch(self, digit_images: List[np.ndarray]) -> List[int]:
        """
        Recognize a batch of digit images efficiently.

        Args:
            digit_images: List of digit images

        Returns:
            List of predicted digits
        """
        # Resize all images
        images_resized = [resize_to_32x32(img, maintain_aspect=True)
                         for img in digit_images]

        # Extract features for all images
        features = self.feature_extractor.extract_features_batch(images_resized)

        # Predict
        predictions = self.classifier.predict(features)

        return predictions.tolist()
