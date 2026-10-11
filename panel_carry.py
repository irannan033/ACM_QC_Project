"""
panel_carry.py  -  detect "a panel is moving WITH the hand"
===========================================================

Problem with the old logic
--------------------------
PanelPresenceDetector compared each area with a calibrated "empty"
background.  In your set-up both areas are already full of fabric, so
"panel present" was true almost all the time and the count simply
followed hand movement.

Idea used here
--------------
Forget the background.  Look only at the zone IN FRONT OF and BESIDE the
palm (never behind the wrist, where the forearm / sleeve is) and ask:

    Is something that is NOT hand-coloured changing/moving there?

An empty hand moving over a pile leaves that zone almost completely
static (score ~0.00-0.05).  A hand pulling or carrying a panel makes a
large part of the zone change (score ~0.4-0.9).

Usage
-----
    carry = CarryDetector()

    # call it every time MediaPipe produced a new result:
    carrying = carry.update(process_frame_bgr, cached_hand_points, video_time)

    # `carrying` stays True for `hold_seconds` after the last positive, so
    # it can be read on every frame.
"""

import cv2
import numpy as np


class CarryDetector:
    def __init__(
        self,
        score_threshold=0.30,
        diff_threshold=12.0,
        zone_radius_hands=1.8,
        behind_wrist_margin=0.15,
        skin_distance=22.0,
        hold_seconds=0.30,
        min_zone_pixels=200
    ):
        self.score_threshold = score_threshold
        self.diff_threshold = diff_threshold
        self.zone_radius_hands = zone_radius_hands
        self.behind_wrist_margin = behind_wrist_margin
        self.skin_distance = skin_distance
        self.hold_seconds = hold_seconds
        self.min_zone_pixels = min_zone_pixels

        self.previous_lab = None
        self.last_positive_time = -1e9
        self.score = 0.0

        self._grid_shape = None
        self._grid_x = None
        self._grid_y = None

    # ------------------------------------------------------------------

    def reset(self):
        self.previous_lab = None
        self.last_positive_time = -1e9
        self.score = 0.0

    def _grid(self, height, width):
        if self._grid_shape != (height, width):
            self._grid_y, self._grid_x = np.mgrid[0:height, 0:width]
            self._grid_shape = (height, width)

        return self._grid_x, self._grid_y

    # ------------------------------------------------------------------

    def hand_zone_score(self, lab, previous_lab, points):
        """
        Fraction of the in-front-of-the-palm zone that is changing and is
        not skin-coloured.  Returns None if it cannot be measured.
        """
        height, width = lab.shape[:2]
        pts = np.asarray(points, dtype=np.float32)

        wrist = pts[0]
        middle_knuckle = pts[9]
        palm = pts[[0, 5, 9, 13, 17]].mean(axis=0)

        hand_size = max(
            float(np.ptp(pts[:, 0])),
            float(np.ptp(pts[:, 1])),
            30.0
        )

        direction = middle_knuckle - wrist
        norm = float(np.linalg.norm(direction))

        if norm < 1e-3:
            return None

        direction = direction / norm
        radius = self.zone_radius_hands * hand_size

        # Work in a crop around the palm (much cheaper than full frame).
        x1 = int(max(0, palm[0] - radius))
        y1 = int(max(0, palm[1] - radius))
        x2 = int(min(width, palm[0] + radius + 1))
        y2 = int(min(height, palm[1] + radius + 1))

        if x2 - x1 < 8 or y2 - y1 < 8:
            return None

        grid_x, grid_y = self._grid(height, width)

        rel_x = grid_x[y1:y2, x1:x2] - palm[0]
        rel_y = grid_y[y1:y2, x1:x2] - palm[1]

        inside = (rel_x * rel_x + rel_y * rel_y) <= radius * radius

        in_front = (
            rel_x * direction[0] + rel_y * direction[1]
            >= -self.behind_wrist_margin * hand_size
        )

        zone = inside & in_front

        # Exclude the hand itself (dilated hull of the landmarks).
        hand_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)

        hull = cv2.convexHull(
            (pts - np.array([x1, y1], dtype=np.float32)).astype(np.int32)
        )

        cv2.fillConvexPoly(hand_mask, hull, 255)

        kernel_size = int(0.35 * hand_size) | 1

        hand_mask = cv2.dilate(
            hand_mask,
            np.ones((kernel_size, kernel_size), np.uint8)
        )

        zone &= hand_mask == 0

        # Exclude skin-coloured pixels (bare forearm, other hand) using
        # the colour of THIS hand's palm.
        palm_samples = []

        for index in (0, 5, 9, 13, 17):
            px = int(np.clip(pts[index][0], 0, width - 1))
            py = int(np.clip(pts[index][1], 0, height - 1))
            palm_samples.append(lab[py, px])

        palm_lab = np.median(np.asarray(palm_samples), axis=0)

        crop_lab = lab[y1:y2, x1:x2]

        skin_like = (
            np.linalg.norm(crop_lab - palm_lab, axis=2)
            < self.skin_distance
        )

        zone &= ~skin_like

        if int(zone.sum()) < self.min_zone_pixels:
            return None

        # Frame-to-frame change inside the zone.
        change = np.linalg.norm(
            crop_lab - previous_lab[y1:y2, x1:x2],
            axis=2
        )

        moving = (change > self.diff_threshold).astype(np.uint8)

        moving = cv2.morphologyEx(
            moving,
            cv2.MORPH_CLOSE,
            np.ones((9, 9), np.uint8)
        )

        return float(moving[zone].mean())

    # ------------------------------------------------------------------

    def update(self, frame_bgr, hand_points_list, now):
        """
        frame_bgr        : the processing-resolution frame
        hand_points_list : list of hands, each a list of 21 (x, y) points
        now              : time in seconds
        Returns True while a panel appears to be carried with a hand.
        """
        lab = cv2.cvtColor(
            cv2.GaussianBlur(frame_bgr, (5, 5), 0),
            cv2.COLOR_BGR2LAB
        ).astype(np.float32)

        previous = self.previous_lab
        self.previous_lab = lab

        best = 0.0

        if previous is not None and previous.shape == lab.shape:
            for points in hand_points_list or []:
                score = self.hand_zone_score(lab, previous, points)

                if score is not None:
                    best = max(best, score)

        self.score = best

        if best >= self.score_threshold:
            self.last_positive_time = now

        return self.carrying(now)

    def carrying(self, now):
        return now - self.last_positive_time <= self.hold_seconds
