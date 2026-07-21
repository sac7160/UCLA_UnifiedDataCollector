# wristpad.gui

The two PyQt windows plus the ring-buffer classes that feed their live plots.

Depends on `wristpad.core` and `wristpad.workers` (for `set_material` /
`toggle_recording`) — `workers` never imports back from here.

| File | What's in it |
|---|---|
| `display_buffers.py` | `ScrollingWaveform` / `ScrollingSpectrogram` / `ScrollingIMU` |
| `instructor_window.py` | `InstructorWindow` |
| `experimenter_window.py` | `ExperimenterWindow` |
