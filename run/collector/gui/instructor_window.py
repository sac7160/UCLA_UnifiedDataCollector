"""
data_collector/gui/instructor_window.py
────────────────────────────────────────────────────────────────────────────
Live waveforms/spectrograms/IMU plots (same as realtime_multimodal_viz.py's
dashboard) plus REC controls, material presets, and stimulus selection.
Reads everything it displays from data_collector.core.state — never
computes anything itself beyond formatting. Its update() is called from
data_collector.py's QTimer tick; it doesn't schedule its own redraws.

Stimulus selection (writing target + "next" button) replaces the old
direct letter-picker dropdown — see _pick_next_item()'s docstring for how
letter/word/sentence map to state.current_label/current_stimulus, and
core/phrase_set.py for where word/sentence text actually comes from.

Every QPushButton here is set to QtCore.Qt.NoFocus. Qt activates a focused
QPushButton on Space/Enter by default — the same physical spacebar press
trial.py's global pynput listener already uses to start/stop REC. If any
button happened to hold keyboard focus (clicking near it, tabbing, etc.),
that same spacebar press would ALSO fire the button's own .clicked() —
harmless for most buttons, but for rec_btn specifically it meant a single
spacebar press could toggle REC twice (once via pynput, once via Qt's own
button activation), silently corrupting trial boundaries. NoFocus removes
every button from the keyboard-focus chain entirely — spacebar reaches
pynput's global listener and nothing else; every button here still works
exactly the same via mouse click.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..core import config, state, phrase_set
from ..core.utils import offset
from ..workers.touch_detection import set_material
from ..workers.trial import toggle_recording
from .display_buffers import ScrollingSpectrogram

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOptions(antialias=False)
pg.setConfigOptions(imageAxisOrder='row-major')


class InstructorWindow(QtWidgets.QMainWindow):
    def __init__(self, window_sec: float, has_camera: bool, use_opengl: bool = False):
        super().__init__()
        self.window_sec = window_sec
        self.has_camera = has_camera
        self._metric_min = None
        self._metric_max = None
        self._last_rec_shown = None   # forces the REC button style to be set on the first update() call
        self._watch_countdown_start_offset = None   # set once, the first time watch data arrives
        self.setWindowTitle('WristPad — Instructor')

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        rec_row = QtWidgets.QHBoxLayout()
        self.rec_btn = QtWidgets.QPushButton('● START RECORDING')
        self.rec_btn.setFocusPolicy(QtCore.Qt.NoFocus)   # see NoFocus note in this file's docstring
        self.rec_btn.setStyleSheet('font-size: 16px; font-weight: bold; padding: 8px; '
                                    'background-color: #d62728; color: white;')
        self.rec_btn.clicked.connect(self._on_rec_clicked)
        rec_row.addWidget(self.rec_btn)
        rec_row.addWidget(QtWidgets.QLabel('  (or press spacebar — press once to start, again to stop)'))
        rec_row.addStretch(1)
        self.watch_countdown_label = QtWidgets.QLabel('watch: waiting for data...')
        self.watch_countdown_label.setStyleSheet('font-size: 14px; font-weight: bold; color: #888; '
                                                   'padding: 2px 10px;')
        rec_row.addWidget(self.watch_countdown_label)
        self.status_label = QtWidgets.QLabel('')
        self.status_label.setStyleSheet('font-size: 12px; color: #333;')
        rec_row.addWidget(self.status_label)
        rec_row.addSpacing(20)
        self.quit_btn = QtWidgets.QPushButton('■ QUIT')
        self.quit_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.quit_btn.setStyleSheet('font-size: 13px; font-weight: bold; padding: 6px 12px; '
                                     'background-color: #444; color: white;')
        self.quit_btn.clicked.connect(self._on_quit_clicked)
        rec_row.addWidget(self.quit_btn)
        outer.addLayout(rec_row)

        meta_row = QtWidgets.QHBoxLayout()
        meta_row.addWidget(QtWidgets.QLabel('surface material:'))
        self.material_label = QtWidgets.QLabel(
            f'[{state.current_material}] {state.touch_band_low_hz:.0f}-{state.touch_band_high_hz:.0f}Hz')
        self.material_label.setStyleSheet('font-weight: bold; color: #1f6feb; padding: 2px 6px; '
                                           'background-color: #eef4ff; border-radius: 3px;')
        for name in config.MATERIAL_PRESETS:
            btn = QtWidgets.QPushButton(name)
            btn.setFocusPolicy(QtCore.Qt.NoFocus)
            btn.clicked.connect(lambda checked=False, n=name: self._on_material_clicked(n))
            meta_row.addWidget(btn)
        meta_row.addWidget(self.material_label)
        meta_row.addSpacing(20)
        outer.addLayout(meta_row)

        # ── condition row: participant / wrist / finger — pure metadata,
        # logged with every trial (see trial.toggle_recording()) but never
        # changes capture behavior; the experimenter arranges the physical
        # setup (wrist lifted vs. resting, which finger) by hand and just
        # keeps this in sync with what's actually happening. ──
        cond_row = QtWidgets.QHBoxLayout()
        cond_row.addWidget(QtWidgets.QLabel('participant:'))
        self.participant_edit = QtWidgets.QLineEdit()
        self.participant_edit.setPlaceholderText('e.g. P1')
        self.participant_edit.setMaximumWidth(70)
        self.participant_edit.setText(state.current_participant)
        # WA_InputMethodEnabled=False turns off IME composition for this
        # widget entirely — the crash reported wasn't from committed
        # non-ASCII text (the validator below already blocks that), it was
        # happening during IME *composition* itself (the "still typing,
        # not yet committed" state a Korean/Japanese/Chinese input method
        # goes through before a character is finalized) — which fires
        # before textChanged and so a validator alone can never catch it.
        # This tells Qt not to route this widget through the system IME at
        # all, so composition never starts here in the first place;
        # keystrokes arrive as plain key events instead. Participant IDs
        # are simple ASCII identifiers, so there's no legitimate reason
        # this field would ever need IME composition anyway.
        self.participant_edit.setAttribute(QtCore.Qt.WA_InputMethodEnabled, False)
        # Restricted to ASCII letters/digits/underscore/hyphen — belt and
        # suspenders alongside the IME cutoff above, in case any other
        # non-ASCII input path (e.g. paste) reaches this field.
        self.participant_edit.setValidator(
            QtGui.QRegExpValidator(QtCore.QRegExp(r'[A-Za-z0-9_-]*')))
        self.participant_edit.textChanged.connect(self._on_participant_changed)
        cond_row.addWidget(self.participant_edit)
        cond_row.addSpacing(16)

        cond_row.addWidget(QtWidgets.QLabel('wrist:'))
        self.wrist_combo = QtWidgets.QComboBox()
        self.wrist_combo.addItems(config.WRIST_CONDITIONS)
        self.wrist_combo.setCurrentText(state.current_wrist_condition)
        self.wrist_combo.currentTextChanged.connect(self._on_wrist_changed)
        cond_row.addWidget(self.wrist_combo)
        cond_row.addSpacing(16)

        cond_row.addWidget(QtWidgets.QLabel('finger:'))
        self.finger_condition_combo = QtWidgets.QComboBox()
        self.finger_condition_combo.addItems(config.FINGER_CONDITIONS)
        self.finger_condition_combo.setCurrentText(state.current_finger_condition)
        self.finger_condition_combo.currentTextChanged.connect(self._on_finger_condition_changed)
        cond_row.addWidget(self.finger_condition_combo)
        cond_row.addStretch(1)
        outer.addLayout(cond_row)

        # ── supplementary task override: checking either one replaces the
        # normal writing-target-based folder label with 'supplementary1'/
        # 'supplementary2' (see _pick_next_item()) and locks in the
        # protocol's fixed conditions for that block (wood surface, word
        # writing target, plus the matching wrist/finger condition) — see
        # _on_supplementary_changed(). Mutually exclusive: checking one
        # unchecks the other. ──
        supp_row = QtWidgets.QHBoxLayout()
        supp_row.addWidget(QtWidgets.QLabel('supplementary:'))
        self.supp1_check = QtWidgets.QCheckBox('Supplementary 1 (wrist fixed)')
        self.supp1_check.setFocusPolicy(QtCore.Qt.NoFocus)   # same spacebar-safety reasoning as every button here
        self.supp1_check.stateChanged.connect(lambda st: self._on_supplementary_changed('supplementary1', st))
        supp_row.addWidget(self.supp1_check)
        self.supp2_check = QtWidgets.QCheckBox('Supplementary 2 (middle finger)')
        self.supp2_check.setFocusPolicy(QtCore.Qt.NoFocus)
        self.supp2_check.stateChanged.connect(lambda st: self._on_supplementary_changed('supplementary2', st))
        supp_row.addWidget(self.supp2_check)
        supp_row.addStretch(1)
        outer.addLayout(supp_row)

        # ── stimulus row: writing target + next button, replacing the old
        # direct letter-picker dropdown. See _pick_next_item(). ──
        stim_row = QtWidgets.QHBoxLayout()
        stim_row.addWidget(QtWidgets.QLabel('writing target:'))
        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.addItems(config.WRITING_TARGETS)
        self.target_combo.setCurrentText(state.current_writing_target)
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        stim_row.addWidget(self.target_combo)

        self.next_btn = QtWidgets.QPushButton('\u21bb next')
        self.next_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.next_btn.setStyleSheet('font-weight: bold; padding: 4px 10px;')
        self.next_btn.clicked.connect(self._on_next_clicked)
        stim_row.addWidget(self.next_btn)

        stim_row.addWidget(QtWidgets.QLabel('to write:'))
        self.class_preview_label = QtWidgets.QLabel('(press next)')
        self.class_preview_label.setStyleSheet('font-weight: bold; font-size: 16px; padding: 2px 8px; '
                                                'background-color: #fffbe6; border-radius: 3px;')
        self.class_preview_label.setWordWrap(True)
        stim_row.addWidget(self.class_preview_label, 1)
        outer.addLayout(stim_row)

        # ── task timer: a plain stopwatch the instructor starts/resets by
        # hand at each block boundary, to track elapsed time against the
        # protocol's "2 min writing, 1 min break" structure. Purely a
        # display — never gates or affects capture. ──
        timer_row = QtWidgets.QHBoxLayout()
        timer_row.addWidget(QtWidgets.QLabel('task timer:'))
        self.task_timer_label = QtWidgets.QLabel('00:00')
        self.task_timer_label.setStyleSheet(
            'font-size: 22px; font-weight: bold; font-family: Menlo, Consolas, monospace; '
            'padding: 4px 14px; background-color: #222; color: #2ecc40; border-radius: 4px;')
        timer_row.addWidget(self.task_timer_label)
        self.task_timer_start_btn = QtWidgets.QPushButton('\u25b6 start')
        self.task_timer_start_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.task_timer_start_btn.clicked.connect(self._on_task_timer_start_clicked)
        timer_row.addWidget(self.task_timer_start_btn)
        self.task_timer_reset_btn = QtWidgets.QPushButton('\u25a0 reset')
        self.task_timer_reset_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.task_timer_reset_btn.clicked.connect(self._on_task_timer_reset_clicked)
        timer_row.addWidget(self.task_timer_reset_btn)
        timer_row.addWidget(QtWidgets.QLabel('  (green <1:30, amber <2:00, red \u22652:00 — writing-block guide only)'))
        timer_row.addStretch(1)
        outer.addLayout(timer_row)

        # Threshold/hysteresis are fixed constants in this calibrated-floor
        # design (config.TOUCH_ON_THRESHOLD_DB / TOUCH_OFF_THRESHOLD_DB) —
        # these two spinboxes are read-only, just showing what's active.
        thr_row = QtWidgets.QHBoxLayout()
        thr_row.addWidget(QtWidgets.QLabel('touch threshold (dB above calibrated floor):'))
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setDecimals(1); self.threshold_spin.setRange(-10.0, 60.0)
        self.threshold_spin.setEnabled(False)
        self.threshold_spin.setValue(state.touch_on_threshold_db)
        thr_row.addWidget(self.threshold_spin)
        thr_row.addWidget(QtWidgets.QLabel('hysteresis (dB):'))
        self.hysteresis_spin = QtWidgets.QDoubleSpinBox()
        self.hysteresis_spin.setDecimals(1); self.hysteresis_spin.setRange(0.0, 30.0)
        self.hysteresis_spin.setEnabled(False)
        self.hysteresis_spin.setValue(state.touch_on_threshold_db - state.touch_off_threshold_db)
        thr_row.addWidget(self.hysteresis_spin)
        thr_row.addStretch(1)
        outer.addLayout(thr_row)

        grid = QtWidgets.QGridLayout()
        outer.addLayout(grid)

        self.pw_surface_wave = self._make_waveform_plot('Surface mic — waveform')
        self.pw_watch_wave   = self._make_waveform_plot('Watch mic — waveform')
        self.pw_surface_spec, self.img_surface_spec = self._make_spec_plot(
            'Surface mic — spectrogram', state.disp_surface_spec)
        self.pw_watch_spec, self.img_watch_spec = self._make_spec_plot(
            'Watch mic — spectrogram', state.disp_watch_spec)
        self.pw_wacc,  self.curves_wacc  = self._make_imu_plot('Watch IMU — acc')
        self.pw_wgyro, self.curves_wgyro = self._make_imu_plot('Watch IMU — gyro')
        self.pw_facc,  self.curves_facc  = self._make_imu_plot(f'Fingertip IMU ({state.display_finger}) — acc')
        self.pw_fgyro, self.curves_fgyro = self._make_imu_plot(f'Fingertip IMU ({state.display_finger}) — gyro')
        self.pw_traj = self._make_traj_plot()
        self.pw_traj.setMaximumHeight(220)   # small — lives in the right-hand column, not a full grid cell
        self.traj_label = QtWidgets.QLabel('index tip: no data yet')
        self.traj_label.setStyleSheet('font-size: 11px; color: #333;')
        self.traj_label.setAlignment(QtCore.Qt.AlignCenter)
        self.traj_label.setWordWrap(True)

        if use_opengl:
            for pw in (self.pw_surface_wave, self.pw_watch_wave, self.pw_wacc,
                       self.pw_wgyro, self.pw_facc, self.pw_fgyro):
                try:
                    pw.useOpenGL(True)
                except Exception:
                    pass

        grid.addWidget(self.pw_surface_wave, 0, 0)
        grid.addWidget(self.pw_surface_spec, 0, 1)
        grid.addWidget(self.pw_watch_wave,   1, 0)
        grid.addWidget(self.pw_watch_spec,   1, 1)
        grid.addWidget(self.pw_wacc,  2, 0)
        grid.addWidget(self.pw_wgyro, 2, 1)
        grid.addWidget(self.pw_facc,  3, 0)
        grid.addWidget(self.pw_fgyro, 3, 1)

        cam_status_text = ('camera: tracking active' if has_camera else '--no-camera specified')
        self.cam_status_label = QtWidgets.QLabel(cam_status_text)
        self.cam_status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cam_status_label.setStyleSheet('background-color: #333; color: white; font-size: 12px; padding: 6px;')

        self.touch_label = QtWidgets.QLabel()
        self.touch_label.setAlignment(QtCore.Qt.AlignCenter)
        font = self.touch_label.font(); font.setPointSize(16); font.setBold(True)
        self.touch_label.setFont(font)
        self.touch_label.setMaximumHeight(70)   # was stretch=1 (filled all remaining space) — now capped short
        self._set_touch_visual(False, -60.0)

        # Terminal-style log panel — mirrors everything printed to stdout
        # (see utils.install_stdout_tee / state.log_lines). Only appends
        # new lines each tick (see update()) rather than resetting the
        # whole widget, so scroll position/selection isn't fought over.
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)   # matches state.log_lines' maxlen — old lines just scroll off
        self.log_view.setStyleSheet('background-color: #111; color: #ddd; font-family: Menlo, Consolas, monospace; '
                                     'font-size: 11px;')
        self._log_last_seq = 0

        right_col = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.cam_status_label)   # fixed height
        right_layout.addWidget(self.pw_traj)             # fixed height (capped above), small trail plot
        right_layout.addWidget(self.traj_label)          # fixed height
        right_layout.addWidget(self.touch_label)         # fixed height (capped above), no stretch
        right_layout.addWidget(self.log_view, 1)          # takes the rest of the vertical space

        minmax_row = QtWidgets.QHBoxLayout()
        self.minmax_label = QtWidgets.QLabel('since reset — min=–  max=–')
        self.minmax_label.setStyleSheet('font-size: 13px; font-weight: bold; color: #222; '
                                         'background-color: #eee; padding: 4px; border-radius: 3px;')
        self.minmax_label.setAlignment(QtCore.Qt.AlignCenter)
        self.reset_minmax_btn = QtWidgets.QPushButton('reset min/max')
        self.reset_minmax_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.reset_minmax_btn.clicked.connect(self._reset_minmax)
        minmax_row.addWidget(self.minmax_label, 1)
        minmax_row.addWidget(self.reset_minmax_btn)
        right_layout.addLayout(minmax_row)

        grid.addWidget(right_col, 0, 2, 4, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1); grid.setColumnStretch(2, 1)
        self.resize(1650, 1050)

        self._pick_next_item()   # show something before the first REC rather than leaving the
                                  # preview blank / stimulus at whatever default state.py set

    # ── panel builders ──
    def _make_waveform_plot(self, title: str) -> pg.PlotWidget:
        pw = pg.PlotWidget(title=title)
        pw.setLabel('bottom', 'time relative to now (s)'); pw.setLabel('left', 'amplitude')
        pw.setXRange(-self.window_sec, 0, padding=0); pw.setYRange(-1.05, 1.05, padding=0)
        pw.showGrid(x=True, y=True, alpha=0.25)
        curve = pw.plot(pen=pg.mkPen('#333333', width=1))
        curve.setDownsampling(auto=True, method='peak'); curve.setClipToView(True)
        pw._curve = curve
        return pw

    def _make_spec_plot(self, title: str, spec: "ScrollingSpectrogram"):
        pw = pg.PlotWidget(title=title)
        pw.setLabel('bottom', 'time relative to now (s)'); pw.setLabel('left', 'frequency (Hz)')
        img = pg.ImageItem()
        try:
            cmap = pg.colormap.get('magma', source='matplotlib')
            img.setLookupTable(cmap.getLookupTable())
        except Exception:
            pass
        freq_max = float(spec.freqs[-1])
        img.setImage(spec.get_image(), autoLevels=True)
        img.setRect(QtCore.QRectF(-self.window_sec, 0, self.window_sec, freq_max))
        pw.addItem(img)
        pw.setXRange(-self.window_sec, 0, padding=0); pw.setYRange(0, freq_max, padding=0)
        return pw, img

    def _make_imu_plot(self, title: str, window_sec: float = 5.0):
        pw = pg.PlotWidget(title=title)
        pw.setLabel('bottom', 'time relative to now (s)'); pw.setLabel('left', 'value')
        pw.setXRange(-window_sec, 0, padding=0); pw.showGrid(x=True, y=True, alpha=0.25)
        pw.addLegend(offset=(5, 5))
        curves = {}
        for axis_name in ('x', 'y', 'z'):
            c = pw.plot(pen=pg.mkPen(config.AXIS_COLORS[axis_name], width=1), name=axis_name)
            c.setDownsampling(auto=True, method='peak'); c.setClipToView(True)
            curves[axis_name] = c
        return pw, curves

    def _make_traj_plot(self) -> pg.PlotWidget:
        pw = pg.PlotWidget(title='Index fingertip trajectory (mic-anchored mm once calibrated, '
                                  'else normalized image-plane coords)')
        pw.setLabel('bottom', 'x'); pw.setLabel('left', 'y')
        pw.showGrid(x=True, y=True, alpha=0.25)
        pw.setAspectLocked(True)
        pw.invertY(True)   # image/mic-plane convention: y increases downward
        trail = pw.plot(pen=pg.mkPen('#9467bd', width=1))
        head = pw.plot(pen=None, symbol='o', symbolSize=8, symbolBrush='#d62728')
        pw._trail = trail
        pw._head = head
        return pw

    # ── button handlers ──
    def _on_rec_clicked(self):
        toggle_recording()

    def _on_quit_clicked(self):
        if state.rec_active:
            reply = QtWidgets.QMessageBox.question(
                self, 'Quit while recording?',
                'A recording is currently in progress. Stop it and quit?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            toggle_recording()   # cleanly stop (and save) the in-progress trial before quitting
        QtWidgets.QApplication.instance().quit()   # triggers aboutToQuit -> the same clean shutdown as Ctrl+C

    def _on_material_clicked(self, name: str):
        set_material(name)
        low, high = config.MATERIAL_PRESETS[name]
        self.material_label.setText(f'[{name}] {low:.0f}-{high:.0f}Hz')
        self._reset_minmax()   # old min/max is meaningless once the band changes

    def _on_participant_changed(self, text: str):
        state.current_participant = text.strip()

    def _on_wrist_changed(self, text: str):
        state.current_wrist_condition = text

    def _on_finger_condition_changed(self, text: str):
        # What the *participant* is instructed to write with — separate
        # from state.display_finger, which only controls which finger's
        # fingertip IMU plot is shown live in this window (a visualization
        # choice, not an experimental condition).
        state.current_finger_condition = text

    def _on_target_changed(self, text: str):
        state.current_writing_target = text
        self._pick_next_item()

    def _on_next_clicked(self):
        self._pick_next_item()

    def _on_supplementary_changed(self, which: str, qt_state: int):
        """which is 'supplementary1' or 'supplementary2' — whichever
        checkbox's stateChanged fired. Checking one enforces the finalized
        protocol's fixed conditions for that block (wood surface, word
        writing target — see the protocol summary's "(only writing word,
        wood)" note) and the matching wrist/finger condition, then locks
        the writing-target combo so it can't drift to letter/sentence by
        accident while a supplementary checkbox is active. Unchecking
        releases that lock but deliberately leaves material/wrist/finger
        wherever they landed — not worth guessing what the experimenter
        wants next."""
        checked = (qt_state == QtCore.Qt.Checked)
        other = self.supp2_check if which == 'supplementary1' else self.supp1_check
        if checked:
            other.blockSignals(True)   # mutually exclusive — set the other's UI state without
            other.setChecked(False)    # re-entering this handler recursively for it
            other.blockSignals(False)
            state.current_supplementary = which
            self._on_material_clicked('wood')
            if which == 'supplementary1':
                self.wrist_combo.setCurrentText('fixed')
            elif which == 'supplementary2':
                self.finger_condition_combo.setCurrentText('middle')
            self.target_combo.setCurrentText('word')
            self.target_combo.setEnabled(False)
        else:
            state.current_supplementary = ''
            self.target_combo.setEnabled(True)
        self._pick_next_item()

    def _pick_next_item(self):
        """Draws a new random stimulus for the current writing target and
        pushes it live to state.current_label/current_stimulus — mirrors
        the old class-picker dropdown's behavior of updating the
        experimenter window immediately, not just at REC-start, so the
        participant sees what to write before recording begins.

        current_label groups the trial into a dataset folder: the literal
        letter for writing_target == 'letter' (dataset/<letter>/, same
        layout the ML pipeline already expects), the writing_target name
        itself ('word'/'sentence') for the other two, OR — overriding
        both — 'supplementary1'/'supplementary2' whenever a supplementary
        checkbox is active (see _on_supplementary_changed()), so those
        trials land in their own folders instead of mixing into
        dataset/word/ alongside main-task data. current_stimulus is always
        the literal text to display/write, in every case — see
        trial.process_trial()'s `content` field for where the
        ground-truth text ends up saved."""
        target = state.current_writing_target
        try:
            content = phrase_set.random_item(target, config.PHRASE_SET_PATH)
        except (FileNotFoundError, ValueError) as e:
            self.class_preview_label.setText('(stimulus unavailable — see log)')
            print(f'[STIMULUS] {e}')
            return

        if state.current_supplementary:
            label = state.current_supplementary
        else:
            label = content if target == 'letter' else target
        state.current_label = label
        state.current_stimulus = content
        self.class_preview_label.setText(content)

    def _on_task_timer_start_clicked(self):
        # Temporary diagnostic — helps pin down a reported "spacebar also
        # starts the task timer" issue: if this print appears in the
        # console right when spacebar is pressed (rather than an actual
        # mouse click on the button), that confirms the button is still
        # reachable via keyboard somehow, despite NoFocus — remove once
        # that's confirmed one way or the other.
        print('[TASK TIMER] start clicked/activated')
        state.task_timer_start = offset()

    def _on_task_timer_reset_clicked(self):
        print('[TASK TIMER] reset clicked/activated')
        state.task_timer_start = None
        self.task_timer_label.setText('00:00')
        self.task_timer_label.setStyleSheet(
            'font-size: 22px; font-weight: bold; font-family: Menlo, Consolas, monospace; '
            'padding: 4px 14px; background-color: #222; color: #2ecc40; border-radius: 4px;')

    def _update_task_timer(self):
        if state.task_timer_start is None:
            return
        elapsed = offset() - state.task_timer_start
        mm, ss = divmod(int(elapsed), 60)
        self.task_timer_label.setText(f'{mm:02d}:{ss:02d}')
        if elapsed < 90:
            color = '#2ecc40'
        elif elapsed < 120:
            color = '#f1c40f'
        else:
            color = '#e74c3c'
        self.task_timer_label.setStyleSheet(
            f'font-size: 22px; font-weight: bold; font-family: Menlo, Consolas, monospace; '
            f'padding: 4px 14px; background-color: #222; color: {color}; border-radius: 4px;')

    def _reset_minmax(self):
        self._metric_min = None
        self._metric_max = None

    # ── per-tick redraw ──
    def _update_waveform(self, pw, waveform):
        x, y = waveform.get_xy()
        pw._curve.setData(x, y)
        if len(y) > 1:
            m = float(np.max(np.abs(y))) * 1.2
            pw.setYRange(-max(m, 0.02), max(m, 0.02), padding=0)

    def _update_spec(self, img, spec):
        img.setImage(spec.get_image(), autoLevels=True)

    def _update_imu(self, pw, curves, imu):
        t, x, y, z = imu.get_series()
        curves['x'].setData(t, x); curves['y'].setData(t, y); curves['z'].setData(t, z)
        if len(t) <= 1:
            return
        allv = np.concatenate([x, y, z])
        finite = allv[np.isfinite(allv)]
        if finite.size == 0:
            return
        lo, hi = float(np.min(finite)), float(np.max(finite))
        pad = max((hi - lo) * 0.15, 1e-3)
        pw.setYRange(lo - pad, hi + pad, padding=0)

    def _update_trajectory(self):
        x, y = state.disp_trajectory.get_xy()
        self.pw_traj._trail.setData(x, y)
        self.pw_traj._head.setData(x[-1:], y[-1:])

        traj = state.disp_trajectory.latest
        if not traj:
            self.traj_label.setText('index tip: no data yet')
            return
        calib_tag = '[calibrated]' if traj.get('calibrated') else '[uncalibrated]'
        if not traj['index_record'].detected:
            self.traj_label.setText(f'index tip: not detected  {calib_tag}')
        elif traj.get('global_xy') is not None:
            gx, gy = traj['global_xy']
            height = traj.get('height_mm')
            txt = f'index tip: x={gx:.1f}mm  y={gy:.1f}mm'
            if height is not None:
                txt += f'  h={height:.1f}mm'
            self.traj_label.setText(f'{txt}  {calib_tag}')
        elif traj.get('x_norm') is not None:
            self.traj_label.setText(
                f'index tip: x_norm={traj["x_norm"]:.3f}  y_norm={traj["y_norm"]:.3f}  {calib_tag}')
        else:
            self.traj_label.setText(f'index tip: no position  {calib_tag}')

    def _set_touch_visual(self, is_on: bool, metric_db: float):
        if state.is_calibrating:
            self.touch_label.setStyleSheet('background-color: #f1c40f; color: black;')
            self.touch_label.setText('CALIBRATING...\nKeep surface quiet')
        elif is_on:
            self.touch_label.setStyleSheet('background-color: #2ca02c; color: white;')
            self.touch_label.setText(f'TOUCH ON\n({metric_db:.1f} dB above floor)')
        else:
            self.touch_label.setStyleSheet('background-color: #d62728; color: white;')
            self.touch_label.setText(f'TOUCH OFF\n({metric_db:.1f} dB above floor)')

    def _update_log_panel(self):
        with state.log_lock:
            new_seq = state.log_seq
            if new_seq == self._log_last_seq:
                return
            n_new = min(new_seq - self._log_last_seq, len(state.log_lines))
            new_lines = list(state.log_lines)[-n_new:] if n_new > 0 else []
            self._log_last_seq = new_seq
        if new_lines:
            self.log_view.appendPlainText('\n'.join(new_lines))
            sb = self.log_view.verticalScrollBar()
            sb.setValue(sb.maximum())   # auto-scroll to the newest line

    def _update_watch_countdown(self, countdown_sec: float = 30.0):
        """Starts a one-shot 30s countdown the moment watch data first
        arrives this session — state.watch_audio_offset / state.imu_offset
        are each set exactly once, at their stream's first packet (see
        writers.py), so the earlier of the two marks "data has started
        coming in from the watch"."""
        if self._watch_countdown_start_offset is None:
            candidates = [o for o in (state.watch_audio_offset, state.imu_offset) if o is not None]
            if not candidates:
                return   # still waiting — leave the "waiting for data..." text as-is
            self._watch_countdown_start_offset = min(candidates)

        remaining = countdown_sec - (offset() - self._watch_countdown_start_offset)
        if remaining > 0:
            self.watch_countdown_label.setText(f'watch: {remaining:4.1f}s')
            color = '#2ca02c' if remaining > 10 else ('#d68910' if remaining > 5 else '#d62728')
        else:
            self.watch_countdown_label.setText('watch: 0.0s')
            color = '#d62728'
        self.watch_countdown_label.setStyleSheet(f'font-size: 14px; font-weight: bold; color: {color}; '
                                                  f'padding: 2px 10px;')

    def update(self):
        self._update_waveform(self.pw_surface_wave, state.disp_surface_wave)
        self._update_waveform(self.pw_watch_wave, state.disp_watch_wave)
        self._update_spec(self.img_surface_spec, state.disp_surface_spec)
        self._update_spec(self.img_watch_spec, state.disp_watch_spec)
        self._update_imu(self.pw_wacc, self.curves_wacc, state.disp_watch_acc)
        self._update_imu(self.pw_wgyro, self.curves_wgyro, state.disp_watch_gyro)
        self._update_imu(self.pw_facc, self.curves_facc, state.disp_finger_acc)
        self._update_imu(self.pw_fgyro, self.curves_fgyro, state.disp_finger_gyro)
        self._update_trajectory()
        self._set_touch_visual(state.touch_on_state, state.touch_metric_db)
        self._update_log_panel()
        self._update_watch_countdown()
        self._update_task_timer()

        # These are display-only (setEnabled(False) at construction), but
        # were never refreshed here before — so switching material updated
        # the *actual* thresholds (via rebuild_touch_band_filter, in the
        # audio worker thread) without ever showing the new values here.
        on_db = state.touch_on_threshold_db
        off_db = state.touch_off_threshold_db
        if self.threshold_spin.value() != on_db:
            self.threshold_spin.setValue(on_db)
        hyst_db = on_db - off_db
        if self.hysteresis_spin.value() != hyst_db:
            self.hysteresis_spin.setValue(hyst_db)

        if np.isfinite(state.touch_metric_db) and not state.is_calibrating:
            self._metric_min = state.touch_metric_db if self._metric_min is None \
                else min(self._metric_min, state.touch_metric_db)
            self._metric_max = state.touch_metric_db if self._metric_max is None \
                else max(self._metric_max, state.touch_metric_db)
        if self._metric_min is not None:
            self.minmax_label.setText(f'since reset — min={self._metric_min:.1f}dB  max={self._metric_max:.1f}dB')
        else:
            self.minmax_label.setText('since reset — min=–  max=–')

        if state.rec_active != self._last_rec_shown:
            self._last_rec_shown = state.rec_active
            if state.rec_active:
                self.rec_btn.setText('■ STOP RECORDING')
                self.rec_btn.setStyleSheet('font-size: 16px; font-weight: bold; padding: 8px; '
                                            'background-color: #2ca02c; color: white;')
            else:
                self.rec_btn.setText('● START RECORDING')
                self.rec_btn.setStyleSheet('font-size: 16px; font-weight: bold; padding: 8px; '
                                            'background-color: #d62728; color: white;')

        if state.is_calibrating:
            self.status_label.setText('STATUS: calibrating noise floor... keep surface quiet.')
        else:
            self.status_label.setText(
                f'surface mic RMS={state.mic_rms:.4f}    floor abs={state.noise_floor_db_abs:.1f}dB    '
                f'touch metric={state.touch_metric_db:.1f}dB    material={state.current_material}  '
                f'[{state.touch_band_low_hz:.0f}-{state.touch_band_high_hz:.0f}Hz]')

    def closeEvent(self, event):
        state.stop_event.set()
        super().closeEvent(event)