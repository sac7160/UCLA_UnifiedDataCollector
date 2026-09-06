"""
list_cameras.py
────────────────────────────────────────────────────────────────────────────
Probes camera indices 0..N and shows a live preview of each one in turn, so
you can visually identify which index corresponds to which physical camera
(e.g. built-in FaceTime camera vs a USB webcam) before passing --camera-index
to index_trajectory_viewer.py / unified_collector_final.py.

macOS/OpenCV don't reliably expose camera names, so this is index-by-index
visual identification rather than a labeled listing.

Usage:
    python list_cameras.py                 # probes indices 0-4
    python list_cameras.py --max-index 8    # probes indices 0-8

Controls (while a preview window is open):
    n / SPACE  -> next index
    q / ESC    -> stop entirely
"""

import argparse

import cv2


def main():
    parser = argparse.ArgumentParser(description='Visually identify camera indices')
    parser.add_argument('--max-index', type=int, default=4,
                         help='highest camera index to probe (default 4)')
    args = parser.parse_args()

    for index in range(args.max_index + 1):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f'[{index}] could not open — skipping')
            cap.release()
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'[{index}] opened OK ({w}x{h}) — showing preview. '
              f'n/SPACE = next, q/ESC = stop')

        while True:
            ok, frame = cap.read()
            if not ok:
                print(f'[{index}] frame read failed — moving on')
                break
            cv2.putText(frame, f'camera index = {index}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow('Camera index probe', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('n'), ord(' ')):
                break
            if key in (ord('q'), 27):
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()

    cv2.destroyAllWindows()
    print('[DONE] probed all indices')


if __name__ == '__main__':
    main()