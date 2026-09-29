from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Any

"""
Thin wrapper for external DeepSORT-like trackers.
Tries deep_sort_realtime first, then Ultralytics StrongSORT if available.
API:
  - update(dets: list[dict], frame_bgr=None) -> list[(track_id:int, det_index:int)]
    dets[i] must have 'bbox':(x1,y1,x2,y2).
"""

class DeepSortTracker:
    def __init__(self, *, max_age: int = 60, iou_thresh: float = 0.15, max_dist: float = 0.4,
                 model: str | None = None, device: str | int | None = None):
        self._impl = None
        self._mode = None
        self._next_fallback_id = 1
        self._iou_thresh = float(iou_thresh)

        # 1) deep_sort_realtime
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            kwargs = dict(max_age=max_age, n_init=1, nn_budget=None, override_track_class=None)
            try:
                kwargs["max_cosine_distance"] = max_dist
                self._impl = DeepSort(**kwargs)
            except TypeError:
                kwargs.pop("max_cosine_distance", None)
                kwargs["max_dist"] = max_dist
                self._impl = DeepSort(**kwargs)
            self._mode = 'dsrt'
            print("[TRACK][backend] deep_sort_realtime (n_init=1, iou<=assign gate)")
            return
        except Exception as e:
            print("[TRACK][backend] deep_sort_realtime load fail:", e)
            self._impl = None

        # 2) StrongSORT (단순 대체)
        try:
            from ultralytics.trackers import strongsort
            self._impl = strongsort.StrongSORT(
                model_weights=model or 'osnet_x0_25_msmt17.pt',
                device=device or 'cpu',
                half=False,
                config=None,
            )
            self._mode = 'strongsort'
            print("[TRACK][backend] ultralytics StrongSORT (fallback)")
            return
        except Exception as e:
            print("[TRACK][backend] StrongSORT load fail:", e)
            self._impl = None
            self._mode = None

    def update(self, dets: List[Dict[str, Any]], frame_bgr=None) -> List[Tuple[int, int]]:
        """
        dets[i]: {'bbox':(x1,y1,x2,y2), 'conf':float?, 'cls':int?}
        반환: [(track_id, det_index)]
        """
        if not dets:
            return []

        # ===== DeepSort Realtime 경로 =====
        if self._mode == 'dsrt':
            # ── 입력 준비 (xywh, conf, cls)
            xywhs, confs, clss = [], [], []
            for d in dets:
                x1, y1, x2, y2 = d['bbox']
                w = max(1, x2 - x1); h = max(1, y2 - y1)
                xywhs.append([x1, y1, w, h])
                confs.append(float(d.get('conf', 0.9)))
                clss.append(int(d.get('cls', 0)))

            xywhs = np.asarray(xywhs, dtype=np.float32)
            confs = np.asarray(confs, dtype=np.float32)
            clss  = np.asarray(clss,  dtype=np.int32)

            tracks = None
            # ── A) 4-positional (xywhs, confs, clss, frame)  ← 구버전
            try:
                tracks = self._impl.update_tracks(xywhs, confs, clss, frame_bgr)
            except TypeError:
                # ── B) 4-args + keyword frame (일부 버전)
                try:
                    tracks = self._impl.update_tracks(xywhs, confs, clss, frame=frame_bgr)
                except TypeError:
                    # ── C) detections(list of tuples), frame=...  ← 다른 브랜치
                    try:
                        # (xywh, conf, cls) 리스트로 변환
                        det_list = [ (xywhs[i], float(confs[i]), int(clss[i])) for i in range(len(xywhs)) ]
                        tracks = self._impl.update_tracks(det_list, frame=frame_bgr)
                    except TypeError:
                        # ── D) detections(list), frame positional
                        det_list = [ (xywhs[i], float(confs[i]), int(clss[i])) for i in range(len(xywhs)) ]
                        tracks = self._impl.update_tracks(det_list, frame_bgr)

            # 트랙 → det 매칭 (IoU 기반 1:1 할당)
            det_boxes = np.array([d['bbox'] for d in dets], dtype=np.float32)  # [N,4]
            trk_boxes, trk_ids = [], []
            for t in (tracks or []):
                try:
                    if not t.is_confirmed():  # 라이브러리 공통 인터페이스
                        continue
                    tb = t.to_ltrb()
                    trk_boxes.append(tb)
                    trk_ids.append(int(t.track_id))
                except Exception:
                    continue

            if len(trk_boxes) == 0 or len(det_boxes) == 0:
                return []

            trk_boxes = np.asarray(trk_boxes, dtype=np.float32)
            iou_mat = _iou_matrix(trk_boxes, det_boxes)
            r, c = linear_sum_assignment(1.0 - iou_mat)  # IoU 최대화

            out = []
            thr = getattr(self, "_iou_thresh", 0.0)
            for ti, di in zip(r, c):
                if iou_mat[ti, di] >= thr:
                    out.append((trk_ids[ti], int(di)))
            out.sort(key=lambda x: x[0])
            return out

        # ===== StrongSORT (간단 대체) =====
        if self._mode == 'strongsort':
            # 필요 시 여기도 실제 strongsort로 매핑 확장 가능.
            out = []
            for i, _ in enumerate(dets):
                out.append((100000 + i, i))
            return out

        # ===== Fallback (no tracker) =====
        out = []
        for i, _ in enumerate(dets):
            out.append((self._next_fallback_id, i))
            self._next_fallback_id += 1
        return out


def _iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    iw = max(0, x2 - x1 + 1); ih = max(0, y2 - y1 + 1)
    inter = iw * ih
    a1 = max(0, b1[2]-b1[0]+1) * max(0, b1[3]-b1[1]+1)
    a2 = max(0, b2[2]-b2[0]+1) * max(0, b2[3]-b2[1]+1)
    denom = a1 + a2 - inter + 1e-6
    return inter / denom

def _iou_matrix(trks: np.ndarray, dets: np.ndarray) -> np.ndarray:
    T, N = trks.shape[0], dets.shape[0]
    M = np.zeros((T, N), dtype=np.float32)
    for ti in range(T):
        tb = trks[ti]
        for di in range(N):
            db = dets[di]
            M[ti, di] = _iou(tb, db)
    return M

def _center_distance(b1, b2):
    cx1 = 0.5 * (b1[0] + b1[2]); cy1 = 0.5 * (b1[1] + b1[3])
    cx2 = 0.5 * (b2[0] + b2[2]); cy2 = 0.5 * (b2[1] + b2[3])
    return ((cx1 - cx2)**2 + (cy1 - cy2)**2) ** 0.5
