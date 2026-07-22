"""
data_collector/core/dataset_classes.py
────────────────────────────────────────────────────────────────────────────
The WatchTouch class list (6 simple strokes + 62 letter/digit classes) and
the mapping from a class name to what should actually appear on screen for
it. Shared by the instructor window's class-picker dropdown and the
experimenter window's big stimulus display, so the two can never disagree
about what a given class name means.
"""

import string

STROKE_CLASSES = [
    'line_horizontal_1',   # left -> right
    'line_horizontal_2',   # right -> left
    'line_vertical_1',     # top -> bottom
    'line_vertical_2',     # bottom -> top
    'circle_cw',           # clockwise
    'circle_ccw',          # counterclockwise
]
UPPER_CLASSES = [f'Upper_{c}' for c in string.ascii_uppercase]   # Upper_A .. Upper_Z
LOWER_CLASSES = [f'Lower_{c}' for c in string.ascii_lowercase]   # Lower_a .. Lower_z
DIGIT_CLASSES = [f'digits_{d}' for d in range(10)]               # digits_0 .. digits_9

ALL_CLASSES = STROKE_CLASSES + UPPER_CLASSES + LOWER_CLASSES + DIGIT_CLASSES

_STROKE_GLYPHS = {
    'line_horizontal_1': '→',
    'line_horizontal_2': '←',
    'line_vertical_1':   '↓',
    'line_vertical_2':   '↑',
    'circle_cw':         '↻',
    'circle_ccw':        '↺',
}


def display_text_for_label(label: str) -> str:
    """Maps a dataset class name to what should actually show on screen.

      - Empty label (nothing selected yet) -> a visible reminder, so a
        blank stimulus display reads as "nothing chosen yet" rather than
        "did this break?".
      - Stroke classes (line_horizontal_1, circle_cw, ...) -> a directional
        arrow/rotation glyph (see _STROKE_GLYPHS) — the class name itself
        isn't something a person can write.
      - Letter/digit classes (Upper_A, Lower_a, digits_0) -> just the
        character itself (A, a, 0).
      - Anything else (a free-text label from some other kind of session)
        -> shown as-is, unchanged.
    """
    if not label:
        return '(no class selected)'
    if label in _STROKE_GLYPHS:
        return _STROKE_GLYPHS[label]
    for prefix in ('Upper_', 'Lower_', 'digits_'):
        if label.startswith(prefix) and len(label) == len(prefix) + 1:
            return label[len(prefix):]
    return label