"""
Deterministic benchmark video generator for SafeTV violence detection engine.
Synthesizes realistic human motion dynamics to evaluate discrimination between
violent physical combat and benign activities (walking, rhythmic exercise).
"""

import math
from typing import List, Tuple
import numpy as np


class SyntheticVideoGenerator:
    """
    Renders deterministic spatiotemporal video sequences with mathematical motion models.
    Zero random stubbing - all patterns follow continuous physics-based motion equations.
    """

    @staticmethod
    def create_canvas(width: int = 160, height: int = 120, bg_color: Tuple[int, int, int] = (40, 44, 52)) -> np.ndarray:
        """Create a solid RGB canvas."""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :] = bg_color
        return canvas

    @staticmethod
    def _draw_entity(frame: np.ndarray, cx: int, cy: int, radius: int, color: Tuple[int, int, int]) -> None:
        """Draw an entity circle on an RGB frame using integer meshgrid."""
        h, w, _ = frame.shape
        y_min, y_max = max(0, cy - radius), min(h, cy + radius + 1)
        x_min, x_max = max(0, cx - radius), min(w, cx + radius + 1)

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    frame[y, x] = color

    @classmethod
    def generate_normal_walking(
        cls,
        num_frames: int = 90,
        fps: float = 30.0,
        width: int = 160,
        height: int = 120
    ) -> List[np.ndarray]:
        """
        Synthesizes two people walking calmly past each other.
        Characterized by smooth, constant linear velocity and gentle gait bobbing.
        """
        frames = []
        for t in range(num_frames):
            frame = cls.create_canvas(width, height, bg_color=(235, 238, 242))

            # Person A: walking left to right
            xa = int(20 + (width - 40) * (t / num_frames))
            ya = int(height // 2 + 4 * math.sin(t * 0.4))
            cls._draw_entity(frame, xa, ya, radius=10, color=(45, 90, 180))

            # Person B: walking right to left
            xb = int(width - 20 - (width - 40) * (t / num_frames))
            yb = int(height // 2 + 15 + 4 * math.cos(t * 0.4))
            cls._draw_entity(frame, xb, yb, radius=10, color=(60, 140, 75))

            frames.append(frame)
        return frames

    @classmethod
    def generate_normal_exercise(
        cls,
        num_frames: int = 90,
        fps: float = 30.0,
        width: int = 160,
        height: int = 120
    ) -> List[np.ndarray]:
        """
        Synthesizes high-energy, regular periodic exercise (e.g. jumping jacks / rhythmic aerobics).
        High motion energy, but strictly periodic and non-chaotic.
        """
        frames = []
        for t in range(num_frames):
            frame = cls.create_canvas(width, height, bg_color=(230, 235, 235))

            # Center fitness subject jumping rhythmically
            phase = t * 0.5
            cx = width // 2
            cy = int(height // 2 + 18 * abs(math.sin(phase)))
            arm_spread = int(14 * abs(math.cos(phase)))

            # Torso
            cls._draw_entity(frame, cx, cy, radius=11, color=(210, 100, 30))
            # Left & right arms moving symmetrically
            cls._draw_entity(frame, cx - 14 - arm_spread, cy - 4, radius=6, color=(210, 100, 30))
            cls._draw_entity(frame, cx + 14 + arm_spread, cy - 4, radius=6, color=(210, 100, 30))

            frames.append(frame)
        return frames

    @classmethod
    def generate_violent_brawl(
        cls,
        num_frames: int = 90,
        fps: float = 30.0,
        width: int = 160,
        height: int = 120
    ) -> List[np.ndarray]:
        """
        Synthesizes physical altercation: rapid irregular collision, erratic high-velocity
        directional changes, sudden strikes and tumbling.
        """
        frames = []
        cx1, cy1 = width // 2 - 15, height // 2
        cx2, cy2 = width // 2 + 15, height // 2

        for t in range(num_frames):
            frame = cls.create_canvas(width, height, bg_color=(220, 220, 225))

            # Non-linear chaotic strike dynamics with sharp impact impulses
            impact = 18 * math.sin(t * 1.8) * math.cos(t * 3.1)
            tumble = 12 * math.sin(t * 2.5)

            pos_x1 = int(np.clip(cx1 + impact, 15, width - 15))
            pos_y1 = int(np.clip(cy1 + tumble, 15, height - 15))

            # Reaction displacement
            pos_x2 = int(np.clip(cx2 - impact * 0.85, 15, width - 15))
            pos_y2 = int(np.clip(cy2 - tumble * 0.9, 15, height - 15))

            # Fast erratic strike limb
            strike_x = int((pos_x1 + pos_x2) / 2 + 12 * math.cos(t * 4.2))
            strike_y = int((pos_y1 + pos_y2) / 2 + 12 * math.sin(t * 4.2))

            cls._draw_entity(frame, pos_x1, pos_y1, radius=12, color=(180, 30, 30))
            cls._draw_entity(frame, pos_x2, pos_y2, radius=12, color=(30, 30, 180))
            cls._draw_entity(frame, strike_x, strike_y, radius=7, color=(230, 180, 20))

            frames.append(frame)
        return frames

    @classmethod
    def generate_timeline_scenario(
        cls,
        total_sec: float = 10.0,
        incident_start_sec: float = 3.5,
        incident_end_sec: float = 6.5,
        fps: float = 30.0,
        width: int = 160,
        height: int = 120
    ) -> Tuple[List[np.ndarray], Tuple[float, float]]:
        """
        Synthesizes a continuous real-world timeline scenario:
        [0.0s - 3.5s]: Normal calm walking
        [3.5s - 6.5s]: Physical altercation (violence incident)
        [6.5s - 10.0s]: Normal dispersal / recovery
        Returns frames and ground truth incident interval (start_sec, end_sec).
        """
        total_frames = int(total_sec * fps)
        start_frame = int(incident_start_sec * fps)
        end_frame = int(incident_end_sec * fps)

        frames = []
        for t in range(total_frames):
            sec = t / fps
            if sec < incident_start_sec or sec >= incident_end_sec:
                # Normal activity
                frame = cls.create_canvas(width, height, bg_color=(235, 238, 242))
                rel_t = t if sec < incident_start_sec else (t - end_frame)
                xa = int(25 + (width - 50) * ((rel_t % 90) / 90.0))
                ya = int(height // 2 + 5 * math.sin(rel_t * 0.3))
                xb = int(width - 25 - (width - 50) * ((rel_t % 90) / 90.0))
                yb = int(height // 2 + 15 + 5 * math.cos(rel_t * 0.3))
                cls._draw_entity(frame, xa, ya, radius=10, color=(45, 90, 180))
                cls._draw_entity(frame, xb, yb, radius=10, color=(60, 140, 75))
            else:
                # Violent brawl
                frame = cls.create_canvas(width, height, bg_color=(220, 220, 225))
                brawl_t = t - start_frame
                impact = 18 * math.sin(brawl_t * 1.8) * math.cos(brawl_t * 3.1)
                tumble = 12 * math.sin(brawl_t * 2.5)

                pos_x1 = int(np.clip(width // 2 - 12 + impact, 15, width - 15))
                pos_y1 = int(np.clip(height // 2 + tumble, 15, height - 15))
                pos_x2 = int(np.clip(width // 2 + 12 - impact * 0.85, 15, width - 15))
                pos_y2 = int(np.clip(height // 2 - tumble * 0.9, 15, height - 15))

                strike_x = int((pos_x1 + pos_x2) / 2 + 12 * math.cos(brawl_t * 4.2))
                strike_y = int((pos_y1 + pos_y2) / 2 + 12 * math.sin(brawl_t * 4.2))

                cls._draw_entity(frame, pos_x1, pos_y1, radius=12, color=(180, 30, 30))
                cls._draw_entity(frame, pos_x2, pos_y2, radius=12, color=(30, 30, 180))
                cls._draw_entity(frame, strike_x, strike_y, radius=7, color=(230, 180, 20))

            frames.append(frame)

        return frames, (incident_start_sec, incident_end_sec)
