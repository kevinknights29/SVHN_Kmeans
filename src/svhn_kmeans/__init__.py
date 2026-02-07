"""
SVHN Recognition System

End-to-end system for Street View House Number recognition using:
- K-means feature learning and beam search (classical approach)
- CNN with STN and multi-task learning (deep learning approach)
- Character Query Transformer (modern transformer-based approach)
"""

# Classical approach (K-means + SVM)
from .data_loader import SVHNDataLoader
from .feature_extractor import KMeansFeatureExtractor
from .classifier import DigitClassifier
from .segmentation import CharacterSegmenter, GeometryModel
from .recognizer import BeamSearchRecognizer, SequenceRecognizer
from .pipeline import SVHNRecognitionPipeline

# Deep learning approach (CNN)
try:
    from .cnn_model import EnhancedSVHNCNN, STN, SEBlock, create_model
    from .cnn_utils import (
        MultiOutputLoss,
        measure_prediction,
        EarlyStopping,
        ModelCheckpoint,
        count_parameters,
    )
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False
    EnhancedSVHNCNN = None
    create_model = None

# Transformer approach (Character Query Transformer)
try:
    from .cqt_model import CharacterQueryTransformer, create_cqt_model
    from .cqt_utils import (
        CQTLoss,
        HungarianMatcher,
        compute_accuracy as cqt_compute_accuracy,
        box_cxcywh_to_xyxy,
        box_xyxy_to_cxcywh,
        generalized_box_iou,
    )
    CQT_AVAILABLE = True
except ImportError:
    CQT_AVAILABLE = False
    CharacterQueryTransformer = None
    create_cqt_model = None

__version__ = "0.1.0"

__all__ = [
    # Classical approach
    'SVHNDataLoader',
    'KMeansFeatureExtractor',
    'DigitClassifier',
    'CharacterSegmenter',
    'GeometryModel',
    'BeamSearchRecognizer',
    'SequenceRecognizer',
    'SVHNRecognitionPipeline',
    # CNN approach
    'EnhancedSVHNCNN',
    'STN',
    'SEBlock',
    'create_model',
    'MultiOutputLoss',
    'measure_prediction',
    'EarlyStopping',
    'ModelCheckpoint',
    'count_parameters',
    'CNN_AVAILABLE',
    # CQT approach
    'CharacterQueryTransformer',
    'create_cqt_model',
    'CQTLoss',
    'HungarianMatcher',
    'cqt_compute_accuracy',
    'box_cxcywh_to_xyxy',
    'box_xyxy_to_cxcywh',
    'generalized_box_iou',
    'CQT_AVAILABLE',
]
