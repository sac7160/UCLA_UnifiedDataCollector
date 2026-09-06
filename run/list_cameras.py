# """
# list_cameras.py
# ────────────────────────────────────────────────────────────────────────────
# Probes camera indices 0..N and shows a live preview of each one in turn, so
# you can visually identify which index corresponds to which physical camera
# (e.g. built-in FaceTime camera vs a USB webcam) before passing --camera-index
# to index_trajectory_viewer.py / unified_collector_final.py.

# macOS/OpenCV don't reliably expose camera names, so this is index-by-index
# visual identification rather than a labeled listing.

# Usage:
#     python list_cameras.py                 # probes indices 0-4
#     python list_cameras.py --max-index 8    # probes indices 0-8

# Controls (while a preview window is open):
#     n / SPACE  -> next index
#     q / ESC    -> stop entirely
# """

# import argparse

# import cv2


# def main():
#     parser = argparse.ArgumentParser(description='Visually identify camera indices')
#     parser.add_argument('--max-index', type=int, default=4,
#                          help='highest camera index to probe (default 4)')
#     args = parser.parse_args()

#     for index in range(args.max_index + 1):
#         cap = cv2.VideoCapture(index)
#         if not cap.isOpened():
#             print(f'[{index}] could not open — skipping')
#             cap.release()
#             continue

#         w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         print(f'[{index}] opened OK ({w}x{h}) — showing preview. '
#               f'n/SPACE = next, q/ESC = stop')

#         while True:
#             ok, frame = cap.read()
#             if not ok:
#                 print(f'[{index}] frame read failed — moving on')
#                 break
#             cv2.putText(frame, f'camera index = {index}', (20, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
#             cv2.imshow('Camera index probe', frame)
#             key = cv2.waitKey(1) & 0xFF
#             if key in (ord('n'), ord(' ')):
#                 break
#             if key in (ord('q'), 27):
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 return

#         cap.release()

#     cv2.destroyAllWindows()
#     print('[DONE] probed all indices')


# if __name__ == '__main__':
#     main()

"""
open_webcam.py
────────────────────────────────────────────────────────────────────────────
카메라를 열어서 화면에 실시간으로 띄우는 간단한 테스트 스크립트.
'q' 키를 누르면 종료.

사용법:
    python open_webcam.py            # 기본 index=1, DSHOW 백엔드
    python open_webcam.py --index 0  # 다른 인덱스로 테스트
    python open_webcam.py --index 1 --backend msmf   # 백엔드 비교용
"""

import argparse
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=1, help="camera index (default: 1)")
    parser.add_argument("--backend", type=str, default="dshow",
                         choices=["dshow", "msmf", "any"],
                         help="capture backend (default: dshow)")
    parser.add_argument("--width", type=int, default=None, help="requested frame width")
    parser.add_argument("--height", type=int, default=None, help="requested frame height")
    parser.add_argument("--fps", type=float, default=None, help="requested fps")
    args = parser.parse_args()

    backend_map = {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "any": cv2.CAP_ANY,
    }
    backend = backend_map[args.backend]

    print(f"[OPEN] index={args.index}, backend={args.backend}")
    cap = cv2.VideoCapture(args.index, backend)

    if not cap.isOpened():
        print("[FAIL] camera did not open — try a different index or backend")
        return

    if args.width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps is not None:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] actual resolution: {actual_w:.0f}x{actual_h:.0f} @ {actual_fps:.1f}fps")
    print("[INFO] press 'q' in the video window to quit")

    frame_count = 0
    fail_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            print(f"[WARN] frame read failed (total fails: {fail_count})")
            if fail_count > 30:
                print("[FAIL] too many consecutive read failures — stopping")
                break
            continue

        frame_count += 1
        cv2.imshow(f"webcam index={args.index} ({args.backend})", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"[DONE] total frames read: {frame_count}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()