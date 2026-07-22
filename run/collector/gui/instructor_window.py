"""
data_collector/gui/instructor_window.py
────────────────────────────────────────────────────────────────────────────
Live waveforms/spectrograms/IMU plots (same as realtime_multimodal_viz.py's
dashboard) plus REC controls, material presets, and the label field. Reads
everything it displays from data_collector.core.state — never computes anything
itself beyond formatting. Its update() is called from data_collector.py's
QTimer tick; it doesn't schedule its own redraws.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ..core import config, state, dataset_classes
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
        self.setWindowTitle('WristPad — Instructor')

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        rec_row = QtWidgets.QHBoxLayout()
        self.rec_btn = QtWidgets.QPushButton('● START RECORDING')
        self.rec_btn.setStyleSheet('font-size: 16px; font-weight: bold; padding: 8px; '
                                    'background-color: #d62728; color: white;')
        self.rec_btn.clicked.connect(self._on_rec_clicked)
        rec_row.addWidget(self.rec_btn)
        rec_row.addWidget(QtWidgets.QLabel('  (or press spacebar — press once to start, again to stop)'))
        rec_row.addStretch(1)
        self.status_label = QtWidgets.QLabel('')
        self.status_label.setStyleSheet('font-size: 12px; color: #333;')
        rec_row.addWidget(self.status_label)
        rec_row.addSpacing(20)
        self.quit_btn = QtWidgets.QPushButton('■ QUIT')
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
            btn.clicked.connect(lambda checked=False, n=name: self._on_material_clicked(n))
            meta_row.addWidget(btn)
        meta_row.addWidget(self.material_label)
        meta_row.addSpacing(20)

        meta_row.addWidget(QtWidgets.QLabel('class to write:'))
        self.class_combo = QtWidgets.QComboBox()
        self.class_combo.addItem('— select —', userData='')
        for cls in dataset_classes.ALL_CLASSES:
            self.class_combo.addItem(cls, userData=cls)
        self.class_combo.setMinimumWidth(160)
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        meta_row.addWidget(self.class_combo)
        self.class_preview_label = QtWidgets.QLabel('')
        self.class_preview_label.setStyleSheet('font-weight: bold; font-size: 16px; padding: 2px 8px;')
        meta_row.addWidget(self.class_preview_label)

        meta_row.addStretch(1)
        outer.addLayout(meta_row)

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
        self.reset_minmax_btn.clicked.connect(self._reset_minmax)
        minmax_row.addWidget(self.minmax_label, 1)
        minmax_row.addWidget(self.reset_minmax_btn)
        right_layout.addLayout(minmax_row)

        grid.addWidget(right_col, 0, 2, 4, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1); grid.setColumnStretch(2, 1)
        self.resize(1650, 1050)

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

    def _on_class_changed(self, index: int):
        """Updates the live label/stimulus immediately — not just at
        REC-start — so the experimenter window shows the right thing to
        write before recording even begins. This directly replaces the
        old --session-label CLI flag / label text field for per-trial
        labeling; --session-label still exists, but now only names the
        session folder (e.g. a participant ID), not what gets written."""
        cls = self.class_combo.itemData(index)
        state.current_label = cls
        state.current_stimulus = cls
        self.class_preview_label.setText(dataset_classes.display_text_for_label(cls) if cls else '')

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