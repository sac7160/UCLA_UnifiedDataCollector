"""
wristpad/gui/experimenter_window.py
────────────────────────────────────────────────────────────────────────────
A second, plain window for a second monitor facing the person writing:
current stimulus in large text, a start/stop banner driven by
state.rec_active, and a free-text instruction line. Its update() is called
from data_collector.py's QTimer tick, same as InstructorWindow's.
"""

from pyqtgraph.Qt import QtCore, QtWidgets

from ..core import state


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
        f = self.stimulus_label.font(); f.setPointSize(140); f.setBold(True)
        self.stimulus_label.setFont(f)
        layout.addWidget(self.stimulus_label, 3)

        self.banner_label = QtWidgets.QLabel('')
        self.banner_label.setAlignment(QtCore.Qt.AlignCenter)
        f2 = self.banner_label.font(); f2.setPointSize(36); f2.setBold(True)
        self.banner_label.setFont(f2)
        layout.addWidget(self.banner_label, 1)

        self.instruction_label = QtWidgets.QLabel('')
        self.instruction_label.setAlignment(QtCore.Qt.AlignCenter)
        f3 = self.instruction_label.font(); f3.setPointSize(18)
        self.instruction_label.setFont(f3)
        self.instruction_label.setStyleSheet('color: #555;')
        self.instruction_label.setWordWrap(True)
        layout.addWidget(self.instruction_label, 1)

        self.resize(900, 700)

    def update(self):
        self.stimulus_label.setText(state.current_stimulus)
        self.instruction_label.setText(state.current_instruction)
        if state.rec_active:
            self.banner_label.setText('WRITING START!')
            self.banner_label.setStyleSheet('color: #2ca02c;')
        else:
            self.banner_label.setText('writing end — please wait')
            self.banner_label.setStyleSheet('color: #888;')
