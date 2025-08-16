# drone_figma_ui_with_tensorboard_fixed.py
# UI with strict thread stop, proper return-to-menu, black training backdrop,
# PyBullet main menu restore, and logger: vertical wheel only; horizontal via dragging slider.

import pygame
from multiprocessing import freeze_support
import threading
import sys
import time
import collections
import numpy as np
import pybullet as p
import pybullet_data
from main import GridDroneEnv
import train as t
import os
import subprocess
import webbrowser
import shutil
import math
import ctypes
import inspect

# ---------- CONFIG ----------
SCREEN_W, SCREEN_H = 600, 600
BACKGROUND = (10, 10, 10)
WHITE = (255, 255, 255)
MUTED = (170, 170, 170)
PANEL = (30, 30, 30)
BUTTON_BG = (0, 0, 0)
SELECTED_BLUE = (51, 66, 165)   # #3342a5
TRAIN_HOVER = (120, 200, 120)
TRAIN_NORMAL = (52, 85, 67)
DISABLED_GRAY = (80, 80, 80)
PROGRESS_GREEN = (10, 120, 10)
LOG_BG = (48, 0, 0)
STOP_HOVER_RED = (150, 0, 0)
TRAIN_BACKDROP = (10, 10, 10)

FPS = 60
GRACE_STOP_SEC = 2.0   # how long we wait for graceful stop before forcing

# Layout rects
TITLE_RECT = pygame.Rect(0, 8, SCREEN_W, 64)
LEFT_PANEL = pygame.Rect(20, 124, 236, 395)
PB_RECT = pygame.Rect(277, 124, 303, 303)
TRAIN_BUTTON_RECT = pygame.Rect(315, 448, 228, 60)

GRID_INPUT_RECT = pygame.Rect(int(109.93), int(184.83), int(133.19), int(20.93))
REFRESH_RECT = pygame.Rect(int(116.26), int(482.08), int(128.39), int(20.93))

strength_positions = [
    pygame.Rect(int(37.89), int(245.61), int(49.73), int(20.93)),
    pygame.Rect(int(89.73), int(245.61), int(49.73), int(20.93)),
    pygame.Rect(int(141.56), int(245.61), int(49.73), int(20.93)),
    pygame.Rect(int(193.4), int(245.61), int(49.73), int(20.93)),
]
timestep_positions = [
    pygame.Rect(int(37.89), int(310.94), int(49.73), int(20.93)),
    pygame.Rect(int(89.73), int(310.94), int(49.73), int(20.93)),
    pygame.Rect(int(141.56), int(310.94), int(49.73), int(20.93)),
    pygame.Rect(int(193.4), int(310.94), int(49.73), int(20.93)),
]
lidar_positions = [
    pygame.Rect(int(37.33), int(376.27), int(49.73), int(20.93)),
    pygame.Rect(int(89.17), int(376.27), int(49.73), int(20.93)),
    pygame.Rect(int(141.01), int(376.27), int(49.73), int(20.93)),
    pygame.Rect(int(192.84), int(376.27), int(49.73), int(20.93)),
]
env_positions = [
    pygame.Rect(int(37.89), int(437.05), int(49.73), int(20.93)),
    pygame.Rect(int(89.73), int(437.05), int(49.73), int(20.93)),
    pygame.Rect(int(141.56), int(437.05), int(49.73), int(20.93)),
    pygame.Rect(int(193.4), int(437.05), int(49.73), int(20.93)),
]

# ---------- Thread kill helpers (last resort) ----------
def _async_raise(tid, exctype):
    """Raise an exception in the threads with id tid"""
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), ctypes.py_object(exctype))
    if res == 0:
        return False
    if res != 1:
        # if it returns >1, we're in a bad state; reset
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)
        return False
    return True

def kill_thread(thread: threading.Thread):
    """Forcefully stop a Python thread by injecting SystemExit."""
    if thread is None:
        return False
    if not thread.is_alive():
        return True
    tid = getattr(thread, "_thread_id", None)
    if tid is None:
        # try to find it
        for k, v in threading._active.items():
            if v is thread:
                tid = k
                break
    if tid is None:
        return False
    return _async_raise(tid, SystemExit)

# ---------- WIDGETS ----------
class InputBox:
    def __init__(self, rect, text="", max_len=64, placeholder=""):
        self.rect = pygame.Rect(rect)
        self.text = str(text)
        self.max_len = max_len
        self.active = False
        self.placeholder = placeholder
        self.font = pygame.font.SysFont("arial", 14)
        self.cursor_visible = True
        self._cursor_timer = 0.0
        self._blink = 0.5

    def handle_event(self, event):
        submitted = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                submitted = True
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                ch = event.unicode
                if ch and len(self.text) < self.max_len:
                    self.text += ch
        return submitted

    def update(self, dt):
        if self.active:
            self._cursor_timer += dt
            if self._cursor_timer >= self._blink:
                self._cursor_timer = 0.0
                self.cursor_visible = not self.cursor_visible
        else:
            self.cursor_visible = False
            self._cursor_timer = 0.0

    def draw(self, surf):
        border = WHITE if self.active else (120,120,120)
        pygame.draw.rect(surf, border, self.rect, 2)
        if self.text == "":
            if self.placeholder:
                ph_surf = self.font.render(self.placeholder, True, MUTED)
                surf.blit(ph_surf, (self.rect.x + 6, self.rect.y + (self.rect.h - ph_surf.get_height()) // 2))
        else:
            txt_surf = self.font.render(self.text, True, WHITE)
            surf.blit(txt_surf, (self.rect.x + 6, self.rect.y + (self.rect.h - txt_surf.get_height()) // 2))
            if self.active and self.cursor_visible:
                cx = self.rect.x + 6 + txt_surf.get_width() + 2
                cy1 = self.rect.y + 4
                cy2 = self.rect.y + self.rect.h - 4
                pygame.draw.line(surf, WHITE, (cx, cy1), (cx, cy2), 2)

class SmallToggle:
    def __init__(self, rect, label, selected=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.selected = selected
        self.font = pygame.font.SysFont("arial", 12)

    def draw(self, surf):
        if self.selected:
            pygame.draw.rect(surf, SELECTED_BLUE, self.rect)
            txt = self.font.render(self.label, True, WHITE)
        else:
            pygame.draw.rect(surf, BUTTON_BG, self.rect)
            pygame.draw.rect(surf, (90,90,90), self.rect, 2)
            txt = self.font.render(self.label, True, MUTED)
        surf.blit(txt, (self.rect.centerx - txt.get_width()//2, self.rect.centery - txt.get_height()//2))

    def clicked(self, pos): return self.rect.collidepoint(pos)

class BigButton:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.font = pygame.font.SysFont("arial", 20)

    def draw(self, surf, disabled=False, hover=False):
        color = DISABLED_GRAY if disabled else (TRAIN_HOVER if hover else TRAIN_NORMAL)
        pygame.draw.rect(surf, color, self.rect)
        pygame.draw.rect(surf, (8,8,8), self.rect, 3)
        txt = self.font.render(self.label, True, WHITE)
        surf.blit(txt, (self.rect.centerx - txt.get_width()//2, self.rect.centery - txt.get_height()//2))

    def clicked(self, pos): return self.rect.collidepoint(pos)

    def return_data(self,timesteps_b, grid_val, strength_status, lidar_rays, env_count, desired_fps):
        self.Data = [
            timesteps_b,
            grid_val,
            strength_status,
            lidar_rays,
            env_count,
            desired_fps,
        ]
        return self.Data

class StopButton:
    def __init__(self, rect, label="Stop training"):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.font = pygame.font.SysFont("arial", 18)

    def set_label(self, new_label):
        self.label = new_label

    def draw(self, surf, hover=False):
        bg = STOP_HOVER_RED if hover else BUTTON_BG
        pygame.draw.rect(surf, bg, self.rect)
        pygame.draw.rect(surf, (8,8,8), self.rect, 2)
        txt = self.font.render(self.label, True, WHITE)
        surf.blit(txt, (self.rect.centerx - txt.get_width()//2, self.rect.centery - txt.get_height()//2))

    def clicked(self, pos): return self.rect.collidepoint(pos)

class Logger:
    """
    Scrollable logger:
    - Vertical: mouse wheel only.
    - Horizontal: ONLY by dragging the bottom slider (no wheel-based horizontal).
    """
    def __init__(self, rect, max_lines=200):
        self.rect = pygame.Rect(rect)
        self.lines = collections.deque(maxlen=max_lines)
        self.font = pygame.font.SysFont("arial", 14)
        self.line_spacing = 2
        self.scroll_lines = 0
        # horizontal scroll (pixels)
        self.scroll_x = 0
        self.scroll_x_max = 0
        self.scroll_track_color = (40, 20, 20)
        self.scroll_thumb_color = (150, 80, 80)
        # dragging state for horizontal slider
        self._dragging_h = False
        self._drag_dx = 0
        self._drag_thumb_rect = pygame.Rect(0,0,0,0)

    def add(self, s):
        for line in str(s).splitlines():
            self.lines.append(line)
        self.scroll_lines = 0
        # keep horizontal where user left it (no forced reset)

    def clear(self):
        self.lines.clear()
        self.scroll_lines = 0
        self.scroll_x = 0

    def handle_event(self, event):
        mpos = pygame.mouse.get_pos()
        over = self.rect.collidepoint(mpos)

        # mouse wheel: vertical only
        if event.type == pygame.MOUSEWHEEL and over:
            if event.y > 0:
                self.scroll_up(event.y)
            elif event.y < 0:
                self.scroll_down(-event.y)

        if event.type == pygame.MOUSEBUTTONDOWN and over:
            if event.button == 1:
                # start dragging if clicked on horizontal thumb
                if self._drag_thumb_rect.collidepoint(mpos):
                    self._dragging_h = True
                    self._drag_dx = mpos[0] - self._drag_thumb_rect.x
            if event.button == 4:
                self.scroll_up(1)
            elif event.button == 5:
                self.scroll_down(1)

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self._dragging_h = False

        if event.type == pygame.MOUSEMOTION and self._dragging_h:
            # drag thumb, update scroll_x accordingly
            bar_w = int(self.rect.w * 0.6)
            bar_h = 6
            bar_x = self.rect.centerx - bar_w//2
            bar_y = self.rect.bottom - 12
            # thumb width based on visible/total
            inner_width = self.rect.w - 16
            max_w = max(1, self._current_visible_max_width())
            thumb_w = max(12, int(bar_w * max(0.05, (inner_width / max_w if max_w>0 else 1))))
            # clamp thumb x
            new_thumb_x = max(bar_x, min(mpos[0] - self._drag_dx, bar_x + bar_w - thumb_w))
            self._drag_thumb_rect = pygame.Rect(new_thumb_x, bar_y, thumb_w, bar_h)
            # map thumb position to scroll_x
            pct = 0 if bar_w - thumb_w == 0 else (new_thumb_x - bar_x) / (bar_w - thumb_w)
            self.scroll_x = int(self.scroll_x_max * pct)

    def _current_visible_max_width(self):
        # compute widest line among currently visible lines
        lines_list = list(self.lines)
        total = len(lines_list)
        lh = self.font.get_height() + self.line_spacing
        max_vis = max(1, self.rect.h // lh)
        start = max(0, total - max_vis - self.scroll_lines)
        end = start + max_vis
        visible = lines_list[start:end]
        max_w = 0
        for ln in visible:
            surf_ln = self.font.render(ln, True, WHITE)
            max_w = max(max_w, surf_ln.get_width())
        return max_w

    def scroll_up(self, steps=1):
        total = len(self.lines)
        max_vis = max(1, self.rect.h // (self.font.get_height() + self.line_spacing))
        max_scroll = max(0, total - max_vis)
        self.scroll_lines = min(max_scroll, self.scroll_lines + int(steps))

    def scroll_down(self, steps=1):
        self.scroll_lines = max(0, self.scroll_lines - int(steps))

    def draw(self, surf):
        prev_clip = surf.get_clip()
        surf.set_clip(self.rect)
        pygame.draw.rect(surf, LOG_BG, self.rect)
        pygame.draw.rect(surf, (60,10,10), self.rect, 2)

        lines_list = list(self.lines)
        total = len(lines_list)
        lh = self.font.get_height() + self.line_spacing
        max_vis = max(1, self.rect.h // lh)
        start = max(0, total - max_vis - self.scroll_lines)
        end = start + max_vis
        visible = lines_list[start:end]

        # compute max width among visible lines to adjust scroll_x_max
        max_w = 0
        rendered = []
        for ln in visible:
            surf_ln = self.font.render(ln, True, WHITE)
            rendered.append((ln, surf_ln))
            max_w = max(max_w, surf_ln.get_width())
        inner_width = self.rect.w - 16  # padding
        self.scroll_x_max = max(0, max_w - inner_width)

        x0 = self.rect.x + 8 - self.scroll_x
        y = self.rect.y + 8
        for _, surf_ln in rendered:
            surf.blit(surf_ln, (x0, y))
            y += lh

        # vertical scrollbar if needed
        if total > max_vis:
            track_w = 8
            track_margin = 4
            track_rect = pygame.Rect(self.rect.right - track_margin - track_w, self.rect.y + 8, track_w, self.rect.h - 16)
            pygame.draw.rect(surf, self.scroll_track_color, track_rect, border_radius=3)
            visible_ratio = max_vis / total
            thumb_h = max(12, int(track_rect.h * visible_ratio))
            max_scroll = total - max_vis
            pos_ratio = 0.0 if max_scroll == 0 else (self.scroll_lines / max_scroll)
            thumb_y = int(track_rect.y + (track_rect.h - thumb_h) * (1-pos_ratio))
            thumb_rect = pygame.Rect(track_rect.x, thumb_y, track_rect.w, thumb_h)
            pygame.draw.rect(surf, self.scroll_thumb_color, thumb_rect, border_radius=3)

        # bottom horizontal slider (drag to scroll horizontally)
        if self.scroll_x_max > 0:
            pct = (self.scroll_x / self.scroll_x_max) if self.scroll_x_max>0 else 0.0
            bar_w = int(self.rect.w * 0.6)
            bar_h = 6
            bar_x = self.rect.centerx - bar_w//2
            bar_y = self.rect.bottom - 12
            pygame.draw.rect(surf, (40,40,40), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
            thumb_w = max(12, int(bar_w * max(0.05, (inner_width / max_w if max_w>0 else 1))))
            thumb_x = int(bar_x + (bar_w - thumb_w) * pct)
            pygame.draw.rect(surf, (120,120,120), (thumb_x, bar_y, thumb_w, bar_h), border_radius=3)
            # store rect for hit-testing / dragging
            self._drag_thumb_rect = pygame.Rect(thumb_x, bar_y, thumb_w, bar_h)
        else:
            self._drag_thumb_rect = pygame.Rect(0,0,0,0)

        surf.set_clip(prev_clip)

# ---------- PyBullet helpers ----------
def build_scene(grid_size, stage, lidar, rr, urdf="drone.urdf"):
    a = GridDroneEnv(
        grid_size=grid_size,
        render = False,
        drone_urdf_path=urdf,
        lidar_num_rays=lidar,
        lidar_max_range=8.0,
        max_vel=4.0,
        hover_z=1,
        timestep=1.0 / rr,
        episode_timeout=20.0,
        draw_lidar=True,
        action_smooth_alpha=0.2,
        stage = stage
    )
    a.reset()
    cx, cy = grid_size * 0.5, grid_size * 0.5
    return cx, cy

# ---------- MAIN ----------
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Drone Simulation Training (thread-stop fixed)")
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("arial", 28)
    label_font = pygame.font.SysFont("arial", 14)
    small_font = pygame.font.SysFont("arial", 12)

    internal_logger = Logger(pygame.Rect(0,0,1,1))
    progress = None

    class ProgressBarLocal:
        def __init__(self, rect):
            self.rect = pygame.Rect(rect)
            self.a = 0.0
            self.b = 1.0
            self.font = pygame.font.SysFont("arial", 16)
        def set_b(self, b):
            try: self.b = float(b) if float(b) > 0 else 1.0
            except: self.b = 1.0
        def set_a(self, a):
            try:
                self.a = max(0.0, float(a))
            except:
                self.a = 0.0
        def draw(self, surf):
            pygame.draw.rect(surf, (0,0,0), self.rect)
            pygame.draw.rect(surf, (80,80,80), self.rect, 2)
            frac = 0.0 if self.b == 0 else min(1.0, self.a / self.b)
            inner_w = int(self.rect.w * frac)
            if inner_w > 0:
                pygame.draw.rect(surf, PROGRESS_GREEN, (self.rect.x, self.rect.y, inner_w, self.rect.h))
            txt = self.font.render(f"Progress {int(frac*100)}%", True, WHITE)
            surf.blit(txt, (self.rect.centerx - txt.get_width()//2, self.rect.centery - txt.get_height()//2))

    progress = ProgressBarLocal(pygame.Rect(178, 534, 260, 32))

    grid_input = InputBox(GRID_INPUT_RECT, text="5", placeholder="")
    refresh_input = InputBox(REFRESH_RECT, text=str(FPS), placeholder="")

    strength_labels = ["Low", "Medium", "Hard", "Auto"]
    # default Medium selected; Auto behaves like Hard(2)
    strength_btns = [SmallToggle(strength_positions[i], strength_labels[i], selected=(strength_labels[i]=="Medium")) for i in range(4)]
    timestep_labels = ["30K", "80K", "1M", ""]
    timestep_btns = [SmallToggle(timestep_positions[i], timestep_labels[i], selected=(timestep_labels[i]=="1M")) for i in range(4)]
    timestep_custom = InputBox((timestep_positions[-1].x, timestep_positions[-1].y, timestep_positions[-1].w, timestep_positions[-1].h), text="", placeholder="Custom")

    lidar_labels = ["30","50","80",""]
    lidar_btns = [SmallToggle(lidar_positions[i], lidar_labels[i], selected=(lidar_labels[i]=="80")) for i in range(4)]
    lidar_custom = InputBox((lidar_positions[-1].x, lidar_positions[-1].y, lidar_positions[-1].w, lidar_positions[-1].h), text="", placeholder="Custom")

    env_labels = ["10","20","30",""]
    env_btns = [SmallToggle(env_positions[i], env_labels[i], selected=(env_labels[i]=="20")) for i in range(4)]
    env_custom = InputBox((env_positions[-1].x, env_positions[-1].y, env_positions[-1].w, env_positions[-1].h), text="", placeholder="Custom")

    strength_status = "Medium"
    strength_stage = 1
    lidar_rays = 80
    desired_fps = FPS
    training_started = False
    newg = 5

    TRAIN_PROGRESS_RECT = pygame.Rect(40, 120, 420, 36)
    STOP_BUTTON_RECT = pygame.Rect(480, 120, 100, 36)
    TRAIN_LOG_RECT = pygame.Rect(20, 170, 560, 410)
    TRAIN_NUM_RECT = pygame.Rect(TRAIN_PROGRESS_RECT.x + 20, TRAIN_PROGRESS_RECT.y + TRAIN_PROGRESS_RECT.h - 90, TRAIN_PROGRESS_RECT.w - 200, TRAIN_PROGRESS_RECT.h-10)
    training_number_input = InputBox(TRAIN_NUM_RECT, text="", placeholder="Enter the training number")

    training_logger = None
    stop_button = StopButton(STOP_BUTTON_RECT)

    # PyBullet initial scene (visible in main menu)
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    grid_val = int(grid_input.text) if grid_input.text.isdigit() else 5
    center_x, center_y = build_scene(grid_val, strength_stage, lidar_rays, desired_fps)

    def open_tensorboard(run_number):
        # Get the directory from which the script is running
        current_dir = os.getcwd()
        
        # Format the command
        logdir = f"./sac_drone_tb/SAC_{run_number}"
        cmd = f'tensorboard --logdir {logdir}'

        # Open CMD in the current directory and run TensorBoard
        subprocess.Popen(
            f'start cmd /K "{cmd}"',
            shell=True,
            cwd=current_dir
        )

    def timestep_b_from_buttons():
        sel = next((b for b in timestep_btns if b.selected), None)
        if not sel: return 30000.0
        if sel.label == "30K": return 30000.0
        if sel.label == "80K": return 80000.0
        if sel.label == "1M": return 1000000.0
        try:
            return float(timestep_custom.text) if timestep_custom.text!="" else 1000000.0
        except:
            return 1000000.0
    def lidar_from_buttons():
        sel = next((b for b in lidar_btns if b.selected), None)
        if not sel: return 40
        if sel.label == "":
            try: return int(lidar_custom.text) if lidar_custom.text!="" else 80
            except: return 80
        return int(sel.label)
    def env_from_buttons():
        sel = next((b for b in env_btns if b.selected), None)
        if not sel: return 10
        if sel.label == "":
            try: return int(env_custom.text) if env_custom.text!="" else 20
            except: return 20
        return int(sel.label)

    timesteps_b = timestep_b_from_buttons()
    lidar_rays = lidar_from_buttons()
    env_count = env_from_buttons()
    progress.set_b(timesteps_b)

    is_training = False
    a_value = 0.0
    train_button = BigButton(TRAIN_BUTTON_RECT, "Train your Model")
    train_start = None
    internal_logger.add("UI ready. Press Train to start demo training.")

    angle = 0.0
    last_time = time.time()

    ui_training_screen = False
    train_thread = None
    stop_requested_by_user = False

    # --- helpers for training thread lifecycle ---
    def request_graceful_stop():
        stats = getattr(t, "TRAINING_STATS", None)
        if stats is not None:
            try:
                if hasattr(stats, "lock"):
                    with stats.lock:
                        setattr(stats, "request_stop", True)
                else:
                    setattr(stats, "request_stop", True)
            except Exception:
                pass

    def force_kill_training_thread():
        nonlocal train_thread
        if train_thread is not None and train_thread.is_alive():
            ok = kill_thread(train_thread)
            return ok
        return True

    def cleanup_after_training():
        nonlocal is_training, ui_training_screen, training_logger, progress, stop_requested_by_user, training_started, train_thread
        is_training = False
        stop_requested_by_user = False
        training_started = False  # restore PyBullet display
        progress.set_a(0.0)
        train_thread = None

    def training_log(msg):
        if training_logger is not None:
            training_logger.add(msg)
        else:
            internal_logger.add(msg)

    def safe_snapshot(stats_obj):
        if stats_obj is None:
            return {"num_steps":0,"goals":0,"collision":0,"stage":0,"last_data":None,"distances_len":0,"request_stop":False}
        try:
            if hasattr(stats_obj, "snapshot"):
                return stats_obj.snapshot()
            else:
                return {
                    "num_steps": getattr(stats_obj, "num_steps", 0),
                    "goals": getattr(stats_obj, "goals", 0),
                    "collision": getattr(stats_obj, "collision", 0),
                    "stage": getattr(stats_obj, "stage", 0),
                    "last_data": getattr(stats_obj, "last_data", None),
                    "distances_len": getattr(stats_obj, "distances_len", 0),
                    "request_stop": getattr(stats_obj, "request_stop", False)
                }
        except Exception:
            return {"num_steps":0,"goals":0,"collision":0,"stage":0,"last_data":None,"distances_len":0,"request_stop":False}

    last_steps = -1  # local tracker to avoid spamming logger

    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now

            mouse_pos = pygame.mouse.get_pos()

            try:
                desired_fps = int(refresh_input.text) if refresh_input.text and refresh_input.text.isdigit() else FPS
            except:
                desired_fps = FPS

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Try graceful stop then force kill
                    request_graceful_stop()
                    deadline = time.time() + GRACE_STOP_SEC
                    while train_thread is not None and train_thread.is_alive() and time.time() < deadline:
                        time.sleep(0.05)
                    if train_thread is not None and train_thread.is_alive():
                        force_kill_training_thread()
                    pygame.quit()
                    try:
                        p.disconnect()
                    except:
                        pass
                    sys.exit()

                grid_sub = grid_input.handle_event(event)
                ref_sub = refresh_input.handle_event(event)
                trainnum_sub = training_number_input.handle_event(event)
                tsub = timestep_custom.handle_event(event)
                lsub = lidar_custom.handle_event(event)
                esub = env_custom.handle_event(event)

                if ui_training_screen and training_logger is not None:
                    training_logger.handle_event(event)

                if grid_sub:
                    try:
                        newg = int(grid_input.text)
                        newg = max(1, newg)
                        center_x, center_y = build_scene(newg, strength_stage, lidar_rays , desired_fps)
                    except Exception:
                        internal_logger.add(f"Invalid grid size: {grid_input.text}")

                if trainnum_sub:
                    numtxt = training_number_input.text.strip()
                    if numtxt == "":
                        internal_logger.add("No training number entered.")
                    else:
                        logdir = os.path.join(".", "sac_drone_tb", f"SAC_{numtxt}")
                        tb_path = shutil.which("tensorboard")
                        if tb_path:
                            try:
                                open_tensorboard(numtxt)
                                webbrowser.open("http://localhost:6007/")
                                internal_logger.add(f"TensorBoard launched for {logdir}. Opened http://localhost:6007/")
                            except Exception as e:
                                internal_logger.add(f"Failed to launch TensorBoard: {e}")
                        else:
                            internal_logger.add("TensorBoard not found on PATH. Install tensorboard to use this feature.")

                if tsub:
                    try:
                        newb = float(timestep_custom.text)
                        if newb > 0:
                            timesteps_b = newb
                            progress.set_b(timesteps_b)
                    except Exception:
                        internal_logger.add(f"Invalid timesteps: {timestep_custom.text}")

                if lsub:
                    try:
                        newl = int(lidar_custom.text)
                        lidar_rays = newl
                    except Exception:
                        internal_logger.add(f"Invalid lidar: {lidar_custom.text}")

                if esub:
                    try:
                        newe = int(env_custom.text)
                        env_count = newe
                    except Exception:
                        internal_logger.add(f"Invalid envs: {env_custom.text}")

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    if ui_training_screen:
                        if stop_button.clicked(pos):
                            # If training is running -> Stop training (then back to menu)
                            if is_training and train_thread is not None and train_thread.is_alive():
                                request_graceful_stop()
                                stop_requested_by_user = True
                                training_log("Stop requested by user. Waiting for graceful stop...")
                                # brief grace period
                                deadline = time.time() + GRACE_STOP_SEC
                                while train_thread.is_alive() and time.time() < deadline:
                                    time.sleep(0.05)
                                if train_thread.is_alive():
                                    training_log("Trainer did not stop, forcing thread kill...")
                                    force_kill_training_thread()
                                # ensure dead
                                deadline = time.time() + 1.0
                                while train_thread.is_alive() and time.time() < deadline:
                                    time.sleep(0.02)
                                cleanup_after_training()
                                ui_training_screen = False
                                internal_logger.add("Returned to main menu.")
                            else:
                                # Not training -> treat as Main Menu button
                                ui_training_screen = False
                                cleanup_after_training()
                                internal_logger.add("Returned to main menu.")
                            continue

                    if not ui_training_screen:
                        for b in strength_btns:
                            if b.clicked(pos):
                                strength_status = b.label
                                if strength_status == "Low":
                                    strength_stage = 0
                                elif strength_status == "Medium":
                                    strength_stage = 1
                                elif strength_status == "Hard":
                                    strength_stage = 2
                                else: 
                                    strength_stage = 3
                                for bb in strength_btns: bb.selected = False
                                b.selected = True
                                center_x, center_y = build_scene(newg, strength_stage, lidar_rays , desired_fps)
                                internal_logger.add(f"Strength -> {b.label}")

                        for b in timestep_btns:
                            if b.clicked(pos):
                                for bb in timestep_btns: bb.selected = False
                                b.selected = True
                                if b.label == "":
                                    timestep_custom.active = True
                                    timestep_custom.text = ""
                                else:
                                    if b.label == "30K":
                                        timesteps_b = 30000.0
                                    elif b.label == "80K":
                                        timesteps_b = 80000.0
                                    elif b.label == "1M":
                                        timesteps_b = 1000000.0
                                    progress.set_b(timesteps_b)
                                    internal_logger.add(f"Timesteps -> {b.label} ({int(timesteps_b)})")

                        for b in lidar_btns:
                            if b.clicked(pos):
                                for bb in lidar_btns: bb.selected = False
                                b.selected = True
                                if b.label == "":
                                    lidar_custom.active = True
                                    lidar_custom.text = ""
                                else:
                                    lidar_rays = int(b.label)
                                    internal_logger.add(f"Lidar -> {lidar_rays}")

                        for b in env_btns:
                            if b.clicked(pos):
                                for bb in env_btns: bb.selected = False
                                b.selected = True
                                if b.label == "":
                                    env_custom.active = True
                                    env_custom.text = ""
                                else:
                                    env_count = int(b.label)
                                    internal_logger.add(f"Envs -> {env_count}")

                        if train_button.clicked(pos):
                            if not is_training:
                                is_training = True
                                ui_training_screen = True
                                training_started = True  # hide PyBullet & show training backdrop
                                train_start = time.time()
                                a_value = 0.0
                                progress.rect = TRAIN_PROGRESS_RECT
                                progress.set_a(0.0)
                                progress.set_b(timesteps_b)
                                training_logger = Logger(TRAIN_LOG_RECT)
                                training_logger.add("Training started.")
                                internal_logger.add("Training started.")

                                # Start training in a background thread, with cooperative stop + hard kill fallback
                                def run_training(env_count, timesteps_b, lidar_rays, desired_fps, strength_stage):
                                    try:
                                        # try to annotate TRAINING_STATS so UI can read env_count and target
                                        for _ in range(50):
                                            stats_tmp = getattr(t, "TRAINING_STATS", None)
                                            if stats_tmp is not None:
                                                try:
                                                    setattr(stats_tmp, "target_timesteps", timesteps_b)
                                                    setattr(stats_tmp, "env_count", env_count)
                                                    setattr(stats_tmp, "request_stop", False)
                                                except Exception:
                                                    pass
                                                break
                                            time.sleep(0.05)
                                        t.train(env_count, timesteps_b, lidar_rays, desired_fps, strength_stage)
                                    except BaseException as e:
                                        # SystemExit from kill_thread lands here too
                                        msg = f"Training thread terminated: {e}"
                                        try:
                                            training_logger.add(msg)
                                        except Exception:
                                            pass
                                        return

                                train_thread = threading.Thread(
                                    target=run_training,
                                    args=(env_count, timesteps_b, lidar_rays, desired_fps, strength_stage),
                                    daemon=True
                                )
                                # store internal id for kill helper
                                def _set_tid(_th):
                                    # CPython sets ident after start; we mirror into _thread_id for kill helper
                                    while _th.ident is None:
                                        time.sleep(0.01)
                                    _th._thread_id = _th.ident
                                train_thread.start()
                                threading.Thread(target=_set_tid, args=(train_thread,), daemon=True).start()
                                stop_requested_by_user = False
                            else:
                                internal_logger.add("Training already running.")

            # update inputs
            grid_input.update(dt)
            refresh_input.update(dt)
            training_number_input.update(dt)
            timestep_custom.update(dt)
            lidar_custom.update(dt)
            env_custom.update(dt)

            if timestep_btns[-1].selected and not timestep_custom.active and timestep_custom.text != "":
                try:
                    val = float(timestep_custom.text)
                    timesteps_b = val
                    progress.set_b(timesteps_b)
                except: pass
            if lidar_btns[-1].selected and not lidar_custom.active and lidar_custom.text != "":
                try:
                    lidar_rays = int(lidar_custom.text)
                except: pass
            if env_btns[-1].selected and not env_custom.active and env_custom.text != "":
                try:
                    env_count = int(env_custom.text)
                except: pass

            # ---------------- POLL TRAINING STATS (non-blocking) ----------------
            if is_training:
                stats_obj = getattr(t, "TRAINING_STATS", None)
                snap = safe_snapshot(stats_obj)

                # estimate environment steps (num_steps * env_count) for progress
                try:
                    envs_for_est = int(getattr(stats_obj, "env_count", env_count) if stats_obj is not None else env_count)
                    if envs_for_est <= 0:
                        envs_for_est = max(1, env_count)
                except Exception:
                    envs_for_est = max(1, env_count)
                est_steps = float(snap.get("num_steps", 0)) * float(envs_for_est)
                # write target back (optional)
                if stats_obj is not None:
                    try:
                        setattr(stats_obj, "target_timesteps", timesteps_b)
                    except Exception:
                        pass
                try:
                    a_value = min(est_steps, float(progress.b))
                    progress.set_a(a_value)
                except Exception:
                    pass

                # live log when step changes
                if snap.get("num_steps", 0) != last_steps:
                    last_steps = snap.get("num_steps", 0)
                    training_log(f"[LIVE] est_steps={int(est_steps)}, steps={snap['num_steps']}, goals={snap['goals']}, collisions={snap['collision']}, stage={snap['stage']}, last_data={snap['last_data']}, distances_len={snap.get('distances_len',0)}")

                # stop label updates
                stop_button.set_label("Stop training" if (train_thread is not None and train_thread.is_alive()) else "Main Menu")

                # training end conditions:
                thread_alive = train_thread.is_alive() if train_thread is not None else False
                if float(est_steps) >= float(progress.b):
                    # reached progress target -> stop thread gracefully then return
                    request_graceful_stop()
                    deadline = time.time() + GRACE_STOP_SEC
                    while train_thread is not None and train_thread.is_alive() and time.time() < deadline:
                        time.sleep(0.02)
                    if train_thread is not None and train_thread.is_alive():
                        force_kill_training_thread()
                    training_log("Training finished (progress target reached).")
                    internal_logger.add("Training finished (progress target reached).")
                    cleanup_after_training()
                    # stay on training screen but "Main Menu" button now visible
                    ui_training_screen = True
                    stop_button.set_label("Main Menu")
                elif (train_thread is not None) and (not thread_alive) and (not stop_requested_by_user):
                    # thread finished on its own
                    training_log("Training thread finished.")
                    internal_logger.add("Training thread finished.")
                    cleanup_after_training()
                    ui_training_screen = True
                    stop_button.set_label("Main Menu")

            # render PyBullet only in main menu
            if not training_started:
                try:
                    angle += 30.0 * dt
                    view = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[center_x, center_y, 0],
                                                            distance=10, yaw=angle, pitch=-30, roll=0, upAxisIndex=2)
                    proj = p.computeProjectionMatrixFOV(fov=60, aspect=PB_RECT.w/PB_RECT.h, nearVal=0.1, farVal=100)
                    img = p.getCameraImage(PB_RECT.w, PB_RECT.h, view, proj)
                    rgba = np.array(img[2], dtype=np.uint8).reshape((PB_RECT.h, PB_RECT.w, 4))
                    rgb = rgba[:, :, :3]
                    pb_surface = pygame.image.frombuffer(rgb.tobytes(), (PB_RECT.w, PB_RECT.h), "RGB")
                except Exception:
                    pb_surface = pygame.Surface((PB_RECT.w, PB_RECT.h))
                    pb_surface.fill((120,120,120))
                p.stepSimulation()
            else:
                pb_surface = None  # hidden during training

            # ---------- DRAW ----------
            screen.fill(BACKGROUND)
            pygame.draw.rect(screen, (18,18,18), TITLE_RECT)
            title_surf = title_font.render("Drone Simulation Training", True, WHITE)
            screen.blit(title_surf, (SCREEN_W//2 - title_surf.get_width()//2, TITLE_RECT.y + 10))

            if not ui_training_screen:
                # MAIN MENU
                pygame.draw.rect(screen, PANEL, LEFT_PANEL)
                pygame.draw.rect(screen, (70,70,70), LEFT_PANEL, 2)
                heading = label_font.render("Simulation Settings", True, WHITE)
                screen.blit(heading, (LEFT_PANEL.x + 70, LEFT_PANEL.y + 10))

                screen.blit(label_font.render("Grid Size", True, WHITE), (GRID_INPUT_RECT.x - 70, GRID_INPUT_RECT.y + 1))
                grid_input.draw(screen)

                screen.blit(label_font.render("Simulation Strength", True, WHITE), (strength_positions[0].x, strength_positions[0].y - 30))
                for b in strength_btns: b.draw(screen)

                screen.blit(label_font.render("Total Timesteps", True, WHITE), (timestep_positions[0].x, timestep_positions[0].y - 30))
                for b in timestep_btns: b.draw(screen)
                if timestep_btns[-1].selected:
                    timestep_custom.draw(screen)

                screen.blit(label_font.render("Lidar Rays", True, WHITE), (lidar_positions[0].x, lidar_positions[0].y - 30))
                for b in lidar_btns: b.draw(screen)
                if lidar_btns[-1].selected:
                    lidar_custom.draw(screen)

                screen.blit(label_font.render("Number of Environments", True, WHITE), (env_positions[0].x, env_positions[0].y - 30))
                for b in env_btns: b.draw(screen)
                if env_btns[-1].selected:
                    env_custom.draw(screen)

                screen.blit(label_font.render("Refresh Rate", True, WHITE), (REFRESH_RECT.x - 80, REFRESH_RECT.y + 1))
                refresh_input.draw(screen)

                pygame.draw.rect(screen, (200,200,200), PB_RECT)
                pygame.draw.rect(screen, (30,30,30), PB_RECT, 3)
                if pb_surface is not None:
                    screen.blit(pb_surface, (PB_RECT.x, PB_RECT.y))

                hover = TRAIN_BUTTON_RECT.collidepoint(pygame.mouse.get_pos()) and (not is_training)
                train_button.draw(screen, disabled=is_training, hover=hover)

            else:
                # TRAINING SCREEN
                # big black backdrop so main menu doesn't show through
                screen.fill(TRAIN_BACKDROP)

                # header
                pygame.draw.rect(screen, (18,18,18), TITLE_RECT)
                screen.blit(title_surf, (SCREEN_W//2 - title_surf.get_width()//2, TITLE_RECT.y + 10))

                # training widgets
                training_number_input.draw(screen)
                note = small_font.render("Training number (tensorboard) below", True, MUTED)
                screen.blit(note, (TRAIN_NUM_RECT.x + 250 , TRAIN_NUM_RECT.y + TRAIN_NUM_RECT.h - 20))
                progress.draw(screen)

                # Stop button label reflects mode
                if is_training and train_thread is not None and train_thread.is_alive():
                    stop_button.set_label("Stop training")
                else:
                    stop_button.set_label("Main Menu")
                stop_hover = stop_button.rect.collidepoint(pygame.mouse.get_pos())
                stop_button.draw(screen, hover=stop_hover)

                if training_logger is not None:
                    training_logger.draw(screen)

            pygame.display.flip()
            clock.tick(desired_fps)

    finally:
        try:
            p.disconnect()
        except:
            pass

if __name__ == "__main__":
    freeze_support()
    main()
