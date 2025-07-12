"""
Input Validator

This module defines the input validator for the AlphaCube application. It includes a Pydantic `Input` class that validates input data for the application's tasks. The input data must adhere to specific format and validation rules.

Validation Rules:

- `format` attribute must be either 'moves' or 'stickers'.
    - For 'moves' format, the `scramble` attribute must consist of valid moves.
    - For 'stickers' format (not implemented), additional validation may be performed.
- `beam_width` attribute must be a positive integer.
- `extra_depths` attribute must be a non-negative integer.

```python title="Example Usage"
input_data = {
    'format': 'moves',
    'scramble': "R U R' U'",
    'beam_width': 256,
    'extra_depths': 2,
    'ergonomic_bias': None
}
validated_input = Input(**input_data)
```

:::note

The ergonomic bias information is optional and not strictly validated.

:::
"""

import json
import re
from typing import Optional

from pydantic import BaseModel, model_validator, validator


class Input(BaseModel):
    """
    Input validator (& preprocessor) for AlphaCube.
    """

    format: str  # The format of the input data. Must be either 'moves' or 'stickers'.
    scramble: list  # The scramble data in the specified format.
    beam_width: int  # The beam width for the task. Must be a positive integer.
    extra_depths: int  # The number of extra depths for the task. Must be a non-negative integer.
    ergonomic_bias: Optional[dict]  # Optional ergonomic bias information (not strictly validated).

    @validator("format")
    @classmethod
    def validate_format(cls, value: str) -> str:
        """
        Validate the input format.

        Args:
            value (str): The format value to be validated.

        Returns:
            str: The validated format value.

        Raises:
            ValueError: If the format is not 'moves' or 'stickers'.
        """
        if value not in {"moves", "stickers"}:
            raise ValueError("Invalid input format. Must be either 'moves' or 'stickers'.")
        return value

    @validator("beam_width")
    @classmethod
    def validate_beam_width(cls, value):
        """
        Validate the beam width.

        Args:
            value: The beam width value to be validated.

        Returns:
            int: The validated beam width value.

        Raises:
            ValueError: If the beam width is not a positive integer.
        """
        if value <= 0:
            raise ValueError("Beam width must be a positive integer.")
        return value

    @validator("extra_depths")
    @classmethod
    def validate_extra_depths(cls, value):
        """
        Validate the extra depths.

        Args:
            value: The extra depths value to be validated.

        Returns:
            int: The validated extra depths value.

        Raises:
            ValueError: If the extra depths value is negative.
        """
        if value < 0:
            raise ValueError("Extra depths must be a non-negative integer.")
        return value

    @model_validator(mode="before")
    @classmethod
    def validate_scramble(cls, values):
        """
        Validate & preprocess the scramble data based on the chosen format.

        Args:
            values (dict): The input values.

        Returns:
            dict: The input values with validated scramble data.

        Raises:
            ValueError: If there are invalid moves in 'scramble' for 'moves' format.
            ValueError: If unexpected center-piece configuration in 'scramble' for 'stickers' format (not implemented).
        """
        # TODO: Also check for potential orientation, permutation, parity

        format = values["format"]
        scramble = values["scramble"]
        if scramble is None:
            raise ValueError("`scramble` cannot be None.")

        if format == "moves":
            if isinstance(scramble, str):
                scramble = scramble.split()
            scramble = [m.replace("2'", "2") for m in scramble]
            invalid_moves = [
                m for m in scramble if not re.match(r"^[UDLRFBudlrfb'2]{1,2}$", m)
            ]  # no wide moves -- yet
            if invalid_moves:
                raise ValueError(
                    f"Invalid move{'s' if len(invalid_moves) > 1 else ''} in `scramble`: {invalid_moves}"
                )
        elif format == "stickers":
            if isinstance(scramble, str):
                scramble = json.loads(scramble.replace("\\\n", ""))
            if isinstance(scramble, dict):
                sticker_colors = [scramble[face] for face in "UDLRBF"]
                # Reset axes if centers are modified
                center_indices = [stickers[4] for stickers in sticker_colors]
                if sorted(center_indices) != list(range(6)):
                    raise ValueError("Unexpected center-piece configuration.")
                # Deconstruct a dict of lists to a flat list
                scramble = sum(sticker_colors, [])

        # Update
        values["scramble"] = scramble

        return values
