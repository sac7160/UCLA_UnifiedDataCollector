"""
data_collector/gui/experimenter_window.py
────────────────────────────────────────────────────────────────────────────
A second, plain window for a second monitor facing the person writing:
the current stimulus filling almost the whole window in large, centered
text, plus a small start/stop banner driven by state.rec_active. Its
update() is called from data_collector.py's QTimer tick, same as
InstructorWindow's.

What actually shows for a given class name (an arrow for strokes, just the
character for letters/digits) is decided by core.dataset_classes.display_text_for_label
— the same mapping the instructor window's class-picker dropdown uses, so
the two can't disagree.
"""

from pyqtgraph.Qt import QtCore, QtWidgets

from ..core import state
from ..core.dataset_classes import display_text_for_label


def _stimulus_font_size(text: str) -> int:
    """Stimulus text ranges from a single character/glyph ("A", "5", "→")
    to a longer placeholder message — one fixed size either overflows the
    short ones' visual impact or wraps/clips the long ones. Scaled by
    length instead, so whatever's showing always fills the space well."""
    n = max(1, len(text))
    if n <= 2:
        return 260
    elif n <= 4:
        return 180
    elif n <= 8:
        return 120
    elif n <= 14:
        return 80
    else:
        return 56


class ExperimenterWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('WristPad — Experimenter')
        central = QtWidgets.QWidget()
        central.setStyleSheet('background-color: white;')
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.stimulus_label = QtWidgets.QLabel('')
        self.stimulus_label.setAlignment(QtCore.Qt.AlignCenter)
        self.stimulus_label.setWordWrap(True)
        f = self.stimulus_label.font(); f.setBold(True)
        self.stimulus_label.setFont(f)
        layout.addWidget(self.stimulus_label, 5)   # dominates the window — this is the whole point

        self.banner_label = QtWidgets.QLabel('')
        self.banner_label.setAlignment(QtCore.Qt.AlignCenter)
        f2 = self.banner_label.font(); f2.setPointSize(28); f2.setBold(True)
        self.banner_label.setFont(f2)
        layout.addWidget(self.banner_label, 1)

        self._last_stimulus_shown = None   # avoid rebuilding the QFont every tick when nothing changed

        self.resize(900, 700)

    def update(self):
        if state.current_stimulus != self._last_stimulus_shown:
            self._last_stimulus_shown = state.current_stimulus
            display_text = display_text_for_label(state.current_stimulus)
            self.stimulus_label.setText(display_text)
            f = self.stimulus_label.font()
            f.setPointSize(_stimulus_font_size(display_text))
            f.setItalic(not bool(state.current_stimulus))   # italic only for the "(no class selected)" placeholder
            self.stimulus_label.setFont(f)
            self.stimulus_label.setStyleSheet('color: black;' if state.current_stimulus else 'color: #aaa;')

        if state.rec_active:
            self.banner_label.setText('WRITING START!')
            self.banner_label.setStyleSheet('color: #2ca02c;')
        else:
            self.banner_label.setText('writing end — please wait')
            self.banner_label.setStyleSheet('color: #888;')