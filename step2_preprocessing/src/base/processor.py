"""
Abstract base class for all data processors.

Defines common interface and validation logic that all processors must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """
    Abstract base class for all data processors.

    All processors (image, structured, text) should inherit from this class
    and implement the required abstract methods.

    This provides:
    - Consistent interface across all processors
    - Configuration validation on initialization
    - Common error handling patterns
    - Easier testing through dependency injection
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize processor with configuration.

        Args:
            config: Configuration dictionary containing processor-specific settings

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.validate_config()
        logger.info(f"{self.__class__.__name__} initialized successfully")

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate processor-specific configuration.

        This method should check that all required configuration keys are present
        and have valid values. Raise ValueError if configuration is invalid.

        Raises:
            ValueError: If configuration is missing required keys or has invalid values
        """
        pass

    @abstractmethod
    def process(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Process input data and return structured output.

        Args:
            *args: Processor-specific positional arguments
            **kwargs: Processor-specific keyword arguments

        Returns:
            Dictionary containing processed data, or None if processing fails

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        pass

    def _handle_error(self, error: Exception, context: str = "") -> None:
        """
        Common error handling logic.

        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
        """
        error_msg = f"{self.__class__.__name__} error"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {type(error).__name__}: {str(error)}"
        logger.error(error_msg)

    def get_config_value(self, *keys, default=None, required=False):
        """
        Safely retrieve nested configuration value.

        Args:
            *keys: Sequence of keys to traverse (e.g., 'image', 'normalize_method')
            default: Default value if key not found
            required: If True, raise ValueError if key not found

        Returns:
            Configuration value or default

        Raises:
            ValueError: If required=True and key not found

        Example:
            >>> processor.get_config_value('image', 'normalize_method', default='minmax')
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                if required:
                    raise ValueError(
                        f"Required configuration key not found: {'.'.join(keys)}"
                    )
                return default
        return value


class ImageProcessor(BaseProcessor):
    """
    Base class for image processors.

    Defines additional interface for image-specific operations.
    """

    @abstractmethod
    def load_and_process(self, image_path, **kwargs):
        """
        Load and process an image from file.

        Args:
            image_path: Path to image file
            **kwargs: Additional processing arguments

        Returns:
            Processed image tensor or None
        """
        pass


class StructuredProcessor(BaseProcessor):
    """
    Base class for structured data processors.

    Defines additional interface for temporal/structured data operations.
    """

    @abstractmethod
    def extract_features(self, subject_id, stay_id, study_time, **kwargs):
        """
        Extract features from structured data.

        Args:
            subject_id: Patient identifier
            stay_id: Stay/encounter identifier
            study_time: Timestamp for feature extraction
            **kwargs: Additional extraction arguments

        Returns:
            Dictionary of extracted features or None
        """
        pass


class TextProcessor(BaseProcessor):
    """
    Base class for text/NLP processors.

    Defines additional interface for text processing operations.
    """

    @abstractmethod
    def process_note(self, note_text: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Process clinical note text.

        Args:
            note_text: Raw clinical note text
            **kwargs: Additional processing arguments

        Returns:
            Dictionary containing processed text features or None
        """
        pass
