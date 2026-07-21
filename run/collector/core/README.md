# watchtouch

WatchTouch data-collection app, split into three subpackages:

| Subpackage | What's in it                                                                               |
| ---------- | ------------------------------------------------------------------------------------------ |
| `core/`    | constants, shared mutable state, small utilities                                           |
| `workers/` | session files, stream writers, touch detection, watch TCP, camera, trial boundaries/saving |
| `gui/`     | the two PyQt windows + their display buffers                                               |

Dependency direction is one-way: **gui → workers → core**. Nothing in
`core/` imports from `workers/` or `gui/`, and nothing in `workers/`
imports from `gui/`.

See `data_collector.py` (the entry point, one level up) for the CLI, and
the module docstring in each file for what that file is responsible for.
