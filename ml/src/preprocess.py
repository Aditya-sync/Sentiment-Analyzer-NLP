import re


def clean_text(text: str) -> str:
    """
    Perform lightweight text cleaning for IMDb reviews.
    """

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace multiple whitespace characters
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text