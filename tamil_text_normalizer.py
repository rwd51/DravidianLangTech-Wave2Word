"""
Tamil Text Normalizer for ASR and Dialect Classification
Standard text cleaning and normalization for Tamil language
"""
import re
import unicodedata


class TamilTextNormalizer:
    """
    Tamil text normalizer following standard practices for ASR preprocessing.
    Handles:
    - Unicode normalization (NFC)
    - Tamil-specific character normalization
    - Whitespace normalization
    - Punctuation handling
    - Number normalization (optional)
    """

    def __init__(self,
                 remove_punctuation: bool = True,
                 normalize_whitespace: bool = True,
                 lowercase_english: bool = True,
                 remove_english: bool = False,
                 normalize_numbers: bool = False):
        """
        Initialize the Tamil text normalizer.

        Args:
            remove_punctuation: Remove punctuation marks
            normalize_whitespace: Normalize multiple spaces to single space
            lowercase_english: Convert English characters to lowercase
            remove_english: Remove English characters entirely
            normalize_numbers: Convert numbers to Tamil numerals (optional)
        """
        self.remove_punctuation = remove_punctuation
        self.normalize_whitespace = normalize_whitespace
        self.lowercase_english = lowercase_english
        self.remove_english = remove_english
        self.normalize_numbers = normalize_numbers

        # Tamil Unicode range: U+0B80 to U+0BFF
        self.tamil_pattern = re.compile(r'[\u0B80-\u0BFF]')

        # Tamil digits mapping
        self.tamil_digits = {
            '0': '\u0BE6', '1': '\u0BE7', '2': '\u0BE8', '3': '\u0BE9',
            '4': '\u0BEA', '5': '\u0BEB', '6': '\u0BEC', '7': '\u0BED',
            '8': '\u0BEE', '9': '\u0BEF'
        }

        # Common punctuation marks to remove
        self.punctuation_pattern = re.compile(
            r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~।॥\u0964\u0965]'
        )

        # Tamil-specific normalization mappings
        # Normalize visarga variants
        self.tamil_normalizations = {
            '\u0B83': '\u0B83',  # Tamil Visarga (Aytham)
        }

        # Zero-width characters to remove
        self.zwc_pattern = re.compile(r'[\u200B-\u200F\u202A-\u202E\uFEFF]')

    def normalize_unicode(self, text: str) -> str:
        """Apply NFC Unicode normalization."""
        return unicodedata.normalize('NFC', text)

    def remove_zero_width_chars(self, text: str) -> str:
        """Remove zero-width Unicode characters."""
        return self.zwc_pattern.sub('', text)

    def normalize_tamil_chars(self, text: str) -> str:
        """
        Apply Tamil-specific character normalizations.
        This handles various forms of the same character.
        """
        for old_char, new_char in self.tamil_normalizations.items():
            text = text.replace(old_char, new_char)
        return text

    def handle_punctuation(self, text: str) -> str:
        """Remove or normalize punctuation."""
        if self.remove_punctuation:
            text = self.punctuation_pattern.sub(' ', text)
        return text

    def handle_english(self, text: str) -> str:
        """Handle English characters based on settings."""
        if self.remove_english:
            # Remove all English alphabets
            text = re.sub(r'[a-zA-Z]', '', text)
        elif self.lowercase_english:
            # Keep English but lowercase it
            text = ''.join(
                c.lower() if c.isascii() and c.isalpha() else c
                for c in text
            )
        return text

    def handle_numbers(self, text: str) -> str:
        """Convert Arabic numerals to Tamil numerals if enabled."""
        if self.normalize_numbers:
            for arabic, tamil in self.tamil_digits.items():
                text = text.replace(arabic, tamil)
        return text

    def normalize_spaces(self, text: str) -> str:
        """Normalize whitespace characters."""
        if self.normalize_whitespace:
            # Replace multiple spaces with single space
            text = re.sub(r'\s+', ' ', text)
            # Strip leading/trailing whitespace
            text = text.strip()
        return text

    def __call__(self, text: str) -> str:
        """
        Apply full normalization pipeline to text.

        Args:
            text: Input Tamil text

        Returns:
            Normalized Tamil text
        """
        if not text:
            return ""

        # Step 1: Unicode normalization (NFC)
        text = self.normalize_unicode(text)

        # Step 2: Remove zero-width characters
        text = self.remove_zero_width_chars(text)

        # Step 3: Tamil-specific character normalization
        text = self.normalize_tamil_chars(text)

        # Step 4: Handle punctuation
        text = self.handle_punctuation(text)

        # Step 5: Handle English characters
        text = self.handle_english(text)

        # Step 6: Handle numbers
        text = self.handle_numbers(text)

        # Step 7: Normalize whitespace
        text = self.normalize_spaces(text)

        return text

    def normalize_batch(self, texts: list) -> list:
        """
        Normalize a batch of texts.

        Args:
            texts: List of Tamil texts

        Returns:
            List of normalized texts
        """
        return [self(text) for text in texts]

    def is_valid_tamil(self, text: str) -> bool:
        """
        Check if text contains Tamil characters.

        Args:
            text: Input text

        Returns:
            True if text contains Tamil characters
        """
        return bool(self.tamil_pattern.search(text))


def create_normalizer(preset: str = "default") -> TamilTextNormalizer:
    """
    Create a normalizer with preset configurations.

    Args:
        preset: One of "default", "strict", "minimal"

    Returns:
        Configured TamilTextNormalizer
    """
    presets = {
        "default": {
            "remove_punctuation": True,
            "normalize_whitespace": True,
            "lowercase_english": True,
            "remove_english": False,
            "normalize_numbers": False
        },
        "strict": {
            "remove_punctuation": True,
            "normalize_whitespace": True,
            "lowercase_english": True,
            "remove_english": True,
            "normalize_numbers": True
        },
        "minimal": {
            "remove_punctuation": False,
            "normalize_whitespace": True,
            "lowercase_english": False,
            "remove_english": False,
            "normalize_numbers": False
        }
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")

    return TamilTextNormalizer(**presets[preset])


if __name__ == "__main__":
    # Test the normalizer
    normalizer = create_normalizer("default")

    test_texts = [
        "இன்னைக்கு என்னங்க ஒரே உப்பசமா இருக்குதுங்க.",
        "அங்க என்ன பண்றிங்க வாங்க விஷ்க்குனு போலாம்.",
        "தம்பி பாருங்க அப்பனாட்டும் நடக்குறான்.",
    ]

    print("Tamil Text Normalizer Test")
    print("=" * 60)
    for text in test_texts:
        normalized = normalizer(text)
        print(f"Original:   {text}")
        print(f"Normalized: {normalized}")
        print("-" * 60)
