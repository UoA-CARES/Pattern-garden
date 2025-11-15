import io
import time
import random
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image as PILImage, ImageDraw

from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.uix.widget import Widget
from kivy.core.image import Image as CoreImage
from kivy.logger import Logger
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDRaisedButton, MDFlatButton
from kivymd.uix.textfield import MDTextField
# KivyMD imports
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.uix.gridlayout import GridLayout

from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton, MDFlatButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.textfield import MDTextField
from kivymd.uix.card import MDCard
import random
from kivy.uix.boxlayout import BoxLayout
from config import style_button
from stimuli import numpy_to_texture, draw_shape
from kivy.uix.anchorlayout import AnchorLayout
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.uix.gridlayout import GridLayout

from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton, MDFlatButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.textfield import MDTextField
from kivymd.uix.card import MDCard
import random
from kivy.uix.boxlayout import BoxLayout
from config import style_button
from stimuli import numpy_to_texture, draw_shape
from kivy.uix.anchorlayout import AnchorLayout
import random
import hashlib

def hash_image(arr: np.ndarray) -> str:
    """Return a quick hash of the image array."""
    return hashlib.md5(arr.tobytes()).hexdigest()
# Optional speech (robot) support
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False
    pyttsx3 = None  # type: ignore

from config import (
    BG, TEXT_COLOR, IMAGE_TILE,
    SAMPLE_PREVIEW_SEC, WARMUP_PREVIEW_SEC,
    style_button
)
from models import Item
from engine import AdaptiveEngine, DifficultyLevel, build_item_bank
from stimuli import (
    NoiseFamily, lsystem_image, draw_shape,
    apply_rotation, apply_occlusion, numpy_to_texture
)
from ui.widgets import ImageButton


# ===== Dementia-friendly constants =====
WARM_BG = (0.98, 0.96, 0.86, 1)  # warm cream, high comfort
MIN_TOUCH = dp(100)              # minimum tap target size
BTN_W = dp(220)
BTN_H = dp(100)
BTN_FONT = "28sp"
LABEL_FONT = "22sp"
INSTR_FONT = "24sp"
FEEDBACK_DELAY_S = 1.8
MAX_TRIALS_DEMENTIA = 60

# Set global background color (prefer warm tone; fallback to BG)
Window.clearcolor = WARM_BG if WARM_BG else BG


class PatternGame(MDBoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", spacing=16, padding=16, **kwargs)

    
        controls = MDBoxLayout(size_hint=(1, None), height=dp(120), spacing=12)

        self.start_btn = MDRaisedButton(
            text="Start", icon="play", font_size=BTN_FONT,
            size_hint=(None, None), width=BTN_W, height=BTN_H,
            on_release=self.start_session
        )
        style_button(self.start_btn, primary=True)

        self.pause_btn = MDRaisedButton(
            text="Break", icon="pause", font_size=BTN_FONT,
            size_hint=(None, None), width=BTN_W, height=BTN_H,
            on_release=self.toggle_pause, disabled=True
        )
        style_button(self.pause_btn, primary=False)

        self.end_btn = MDRaisedButton(
            text="Stop", icon="stop", font_size=BTN_FONT,
            size_hint=(None, None), width=BTN_W, height=BTN_H,
            on_release=self.end_session, disabled=True
        )
        style_button(self.end_btn, primary=False)

        self.dialog = None

        self.save_btn = MDRaisedButton(
            text="Save", icon="content-save", font_size=BTN_FONT,
            size_hint=(None, None), width=BTN_W, height=BTN_H,
            on_release=self.show_save_dialog, disabled=True
        )
        style_button(self.save_btn, primary=False)

    


        self.repeat_btn = MDRaisedButton(
            text="Repeat", icon="repeat", font_size=BTN_FONT,
            size_hint=(None, None), width=BTN_W, height=BTN_H,
            on_release=lambda *_: self._repeat_instruction(), disabled=True
        )
        style_button(self.repeat_btn, primary=False)

        # Status panel (left-aligned, large text)
        status_panel = MDBoxLayout(orientation='vertical', size_hint=(1, None), height=dp(120))
        self.status_label = Label(text="Ready.", halign="left", valign="middle", font_size=INSTR_FONT, color=TEXT_COLOR)
        self.session_info = Label(text="Trials: 0", halign="left", valign="middle", font_size=LABEL_FONT, color=TEXT_COLOR)
        status_panel.add_widget(self.status_label)
        status_panel.add_widget(self.session_info)

        # Add controls
        controls.add_widget(self.start_btn)
        controls.add_widget(self.pause_btn)
        controls.add_widget(self.end_btn)
        controls.add_widget(self.save_btn)
        controls.add_widget(self.repeat_btn)
        controls.add_widget(status_panel)
        self.add_widget(controls)

        # Persistent instruction area
        self._stage_instruction = Label(
            text="Press Start to begin.", size_hint=(1, None), height=dp(72),
            font_size=INSTR_FONT, color=TEXT_COLOR
        )
        self.add_widget(self._stage_instruction)

        # Dynamic stage
        self.stage = MDBoxLayout(orientation="vertical", spacing=12, padding=12)
        self.add_widget(self.stage)

        # State
        self.session_active = False
        self.is_paused = False
        self.engine = AdaptiveEngine()
        self.item_bank = build_item_bank(n_items=90, seed=42)
        self.anchors = self._select_anchor_items(3)
        self.probe_family = "lsystem"
        self.trial_log: List[Dict] = []
        self.current_item: Optional[Item] = None
        self.trial_start_time: Optional[float] = None
        self.trial_idx = 0
        self.max_trials = MAX_TRIALS_DEMENTIA
        self.difficulty = DifficultyLevel()

        # phase 0 = warmup, 1 = sample-2AFC, 2 = grid-find
        self.phase = 0

        # Optional TTS (slower, clearer)
        self.tts = pyttsx3.init() if TTS_AVAILABLE else None
        if self.tts:
            try:
                self.tts.setProperty('rate', 150)
                self.tts.setProperty('volume', 0.9)
            except Exception:
                pass

        # store last computed tile size
        self._computed_tile = (IMAGE_TILE[0], IMAGE_TILE[1])



    # ===== Helpers =====
    def _speak(self, text: str):
        if self.tts and text:
            try:
                self.tts.stop()
                self.tts.say(text)
                self.tts.runAndWait()
            except Exception:
                pass

    def show_instruction(self, text: str, speak: bool = True):
        """Update the persistent instruction area and optionally TTS."""
        self._stage_instruction.text = text
        if speak:
            self._speak(text)

    def _repeat_instruction(self):
        self._speak(self._stage_instruction.text)

    def _feedback(self, correct: bool):
        msg = "Correct — well done." if correct else "Not quite — let's try the next one."
        self.show_instruction(msg, speak=True)
        if correct:
            try:
                self.perform_dance()
            except Exception as e:
                print("Dance could not start:", e)

    def _select_anchor_items(self, k: int) -> List[int]:
        rng = np.random.RandomState(123)
        indices = rng.choice(len(self.item_bank), size=k, replace=False)
        return list(map(int, indices))

    def perform_dance(self):
        try:
            # Define a playful dance sequence (timing in milliseconds)
            sequence = [
                (0,    lambda: hardware.move_motor("1", -50, duration=1)),
                (0,    lambda: hardware.move_motor("2", -50, duration=0.5)),

                (600,  lambda: hardware.move_motor("1", 50, duration=0.5)),
                (600,  lambda: hardware.move_motor("2", 50, duration=1)),

                (1200, lambda: hardware.move_motor("1", 50, duration=0.5)),
                (1200, lambda: hardware.move_motor("2", -50, duration=0.5)),

                (1800, lambda: hardware.move_motor("1", -50, duration=0.5)),
                (1800, lambda: hardware.move_motor("2", 50, duration=0.5)),

                (2400, lambda: hardware.move_motor("1", -50, duration=0.5)),
                (2400, lambda: hardware.move_motor("2", 50, duration=0.5)),

                (3000, lambda: hardware.move_motor("1", 50, duration=0.5)),
                (3000, lambda: hardware.move_motor("2", -50, duration=0.5)),

                (3600, lambda: hardware.stop_motor("1")),
                (3600, lambda: hardware.stop_motor("2")),
            ]

            # Schedule each movement with a delay
            for delay_ms, action in sequence:
                QTimer.singleShot(delay_ms, action)

        except Exception as e:
            print("Dance error:", e)
            # Stop motors safely in case of failure
            hardware.stop_motor("1")
            hardware.stop_motor("2")


    def start_session(self, *args):
        """Start the session, but first prompt for Participant ID if not set."""
        if not hasattr(self, "participant_id"):
            # --- Show ID dialog ---
            if self.dialog:
                self.dialog.dismiss()

            self.participant_field = MDTextField(
                hint_text="Enter Participant ID (e.g., 001)",
                helper_text="Use de-identified code only",
                required=True,
                font_size="22sp",
                size_hint_x=1,
                input_filter="int",     # restrict input to numbers
                input_type="number"     # numeric keypad on touchscreens
            )

            self.dialog = MDDialog(
                title="Participant ID",
                type="custom",
                content_cls=self.participant_field,
                buttons=[
                    MDFlatButton(
                        text="CANCEL", font_size="20sp",
                        on_release=lambda x: self.dialog.dismiss()
                    ),
                    MDRaisedButton(
                        text="START", font_size="20sp",
                        on_release=self.confirm_id_and_start
                    ),
                ],
                auto_dismiss=False,
            )
            self.dialog.open()
            return  # don’t start yet, wait for ID
        # --- If ID is already set, go straight into session ---
        self._begin_session()


    def confirm_id_and_start(self, *args):
        """Confirm entered ID and begin session."""
        pid = self.participant_field.text.strip()
        if not pid:
            self.status_label.text = "Participant ID required."
            return

        self.participant_id = pid  # 🔹 save permanently
        self.dialog.dismiss()
        self._begin_session()


    def _begin_session(self):
        """Actual session start logic (split so ID dialog can lead here)."""
        self.session_active = True
        self.is_paused = False
        self.trial_log = []
        self.engine = AdaptiveEngine(init_theta=0.0, lr=0.25)
        self.trial_idx = 0
        self.phase = 0
        self.start_btn.disabled = True
        self.pause_btn.disabled = False
        self.end_btn.disabled = False
        self.save_btn.disabled = True
        self.repeat_btn.disabled = False
        self.status_label.text = "Session: running"
        self.session_info.text = f"Trial {self.trial_idx}/{self.max_trials}"
        self.show_instruction("Warmup — look at the shape, then tap the match.")
        self._speak("Welcome. Let's begin.")
        Clock.schedule_once(lambda dt: self.next_trial(), 0.6)




    def toggle_pause(self, *args):
        if not self.session_active:
            return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.show_instruction("Paused — take a short break. Tap Break again to continue.")
            self.status_label.text = "Session: paused"
        else:
            self.show_instruction("Resuming. Focus on the next screen.")
            self.status_label.text = "Session: running"
            Clock.schedule_once(lambda dt: self.next_trial(), 0.4)

    def end_session(self, *args):
        self.session_active = False
        self.start_btn.disabled = False
        self.pause_btn.disabled = True
        self.end_btn.disabled = True
        self.save_btn.disabled = True   # disable manual save, since auto-save runs
        self.repeat_btn.disabled = True
        self.status_label.text = f"Session ended — {len(self.trial_log)} trials."
        self.show_instruction("Session ended. Results have been saved automatically.")
        self._speak("Session complete. Thank you.")

        # --- Auto-save with participant ID or fallback name ---
        participant_id = getattr(self, "participant_field", None)
        pid = participant_id.text.strip() if participant_id and participant_id.text.strip() else "Anon"

        df = pd.DataFrame(self.trial_log)
        filename = f"P{pid}_results.csv"
        df.to_csv(filename, index=False)

        # --- Show completion dialog ---
        self.dialog = MDDialog(
            title="Session Complete",
            text=f"Great job! You have finished all {len(self.trial_log)} trials.\n\n"
                f"Your results have been saved as:\n{filename}",
            buttons=[
                MDRaisedButton(
                    text="OK",
                    on_release=lambda x: self.dialog.dismiss()
                )
            ],
            auto_dismiss=False,
        )
        self.dialog.open()


    def show_save_dialog(self, *args):
            """Show popup to enter participant code (P001, P002, etc.)"""
            if self.dialog:
                self.dialog.dismiss()

            self.participant_field = MDTextField(
                hint_text="Enter Participant ID (e.g., 001)",
                helper_text="Use de-identified code only",
                required=True,
                size_hint_x=0.8
            )

            self.dialog = MDDialog(
                title="Save Results",
                type="custom",
                content_cls=self.participant_field,
                buttons=[
                    MDFlatButton(text="CANCEL", on_release=lambda x: self.dialog.dismiss()),
                    MDRaisedButton(text="SAVE", on_release=self.confirm_save)
                ],
            )
            self.dialog.open()
    def confirm_save(self, *args):
        """Triggered when SAVE is pressed in dialog"""
        participant_id = self.participant_field.text.strip()
        if not participant_id:
            self.status_label.text = "Participant ID required."
            return

        self.dialog.dismiss()
        self.save_results(participant_id)

    def save_results(self, participant_id, *args):
        """Save with de-identified filename"""
        if not self.trial_log:
            self.status_label.text = "No trials to save."
            return

        df = pd.DataFrame(self.trial_log)
        filename = f"P{participant_id}_results.csv"
        df.to_csv(filename, index=False)

        self.status_label.text = f"Saved: {filename}"
        self.save_btn.disabled = True
        self.show_instruction("Results saved. You may close the app.")
        
    # ===== Orchestrator =====
    def next_trial(self):
        if not self.session_active or self.is_paused:
            return
        if self.trial_idx >= self.max_trials:
            self.end_session()
            return

        # --- Adaptive phase control ---
        window_size = 10
        recent_trials = self.trial_log[-window_size:]
        if len(recent_trials) >= window_size:
            acc = sum(1 for t in recent_trials if t["correct"]) / window_size

            # allow progression up to phase 5 (3×3 grid)
            if acc >= 0.7 and self.phase < 5:
                self.phase += 1  # advance one step
            elif acc < 0.4 and self.phase > 0:
                self.phase -= 1  # step back if struggling

        self.trial_idx += 1
        self.trial_start_time = time.time()
        self.session_info.text = f"Trials: {self.trial_idx}/{self.max_trials}"

        # --- Item selection (IRT/probe/anchor) ---
        candidates = random.sample(self.item_bank, k=min(12, len(self.item_bank)))
        probe_flag = False
        anchor_flag = False
        if random.random() < 0.08:
            fam = self.probe_family
            pool = [it for it in self.item_bank if it.family == fam]
            current_item = random.choice(pool) if pool else self.engine.select_item(candidates)
            probe_flag = True
        elif random.random() < 0.06:
            current_item = self.item_bank[random.choice(self.anchors)]
            anchor_flag = True
        else:
            current_item = self.engine.select_item(candidates)

        self.current_item = current_item

        # --- Route trial types by phase (gentle progression) ---
        if self.phase == 0:
            # Warmup with very simple shapes
            self.run_warmup_2afc(current_item, probe_flag, anchor_flag)

        elif self.phase == 1:
            # Sample 2AFC, slightly harder than warmup
            self.run_sample_2afc(current_item, probe_flag, anchor_flag)

        elif self.phase == 2:
            # 3AFC (intermediate step)
            self.run_multi_afc(current_item, probe_flag, anchor_flag, n_choices=3)

        elif self.phase == 3:
            # 4AFC (harder but still not grid)
            self.run_multi_afc(current_item, probe_flag, anchor_flag, n_choices=4)

        elif self.phase == 4:
            # 2×2 grid (now supports shapes, l-systems, noise)
            self.run_grid_find_2x2(current_item, probe_flag, anchor_flag)

        else:
            # Full 3×3 grid (ramps difficulty inside function)
            self.run_grid_find(current_item, probe_flag, anchor_flag)

    def run_warmup_2afc(self, item, probe: bool, anchor: bool):
            """Warmup 2AFC with dementia-friendly card UI and big centered preview."""


            self.stage.clear_widgets()
            self.status_label.text = f"Trial {self.trial_idx}/{self.max_trials} — Warmup"

            shapes = ["square", "circle", "triangle", "star",  "diamond","rectangle","oval","pentagon","hexagon","cross","arrow"]
            s_correct, s_foil = random.sample(shapes, 2)
            seed = item.base_seed

            preview_arr = draw_shape(s_correct, size=360, seed=seed)
            self.show_instruction("Look carefully. This will disappear. Then tap the same one.")

            preview_card = MDCard(
                orientation="vertical",
                size_hint=(None, None),
                width=dp(340),
                height=dp(340),
                elevation=8,
                shadow_softness=1,
                shadow_offset=(3, -3),
                md_bg_color=(0.98, 0.96, 0.9, 1),
                padding=20,
                radius=[20, 20, 20, 20],
            )

            preview_img = Image(
                texture=numpy_to_texture(preview_arr),
                size_hint=(1, 1),
                allow_stretch=True,
                keep_ratio=True,
            )

            preview_card.add_widget(preview_img)

            # Center it on stage
            preview_box = AnchorLayout(
                anchor_x="center",
                anchor_y="center",
                size_hint=(1, None),
                height=dp(400),
            )
            preview_box.add_widget(preview_card)

            self.stage.add_widget(preview_box)
            
            def show_choices(_dt):
                if not self.session_active or self.is_paused:
                    return
                self.stage.clear_widgets()
                self.show_instruction(f"Which one is the correct shape?")

                row = MDBoxLayout(
                    orientation="vertical",
                    size_hint=(1, None),
                    height=dp(320),
                    padding=20,
                    spacing=20,
                )

                img_correct = draw_shape(s_correct, size=200, seed=seed)
                img_foil = draw_shape(s_foil, size=200, seed=seed + 3)
                choices = [img_correct, img_foil]
                labels = [s_correct, s_foil]

                if random.random() < 0.5:
                    choices.reverse()
                    labels.reverse()

                self._inject_choice_cards(
                    row,
                    choices,
                    correct_idx=labels.index(s_correct),
                    handler=lambda ok: self._handle_2afc_response(
                        ok,
                        item,
                        "warmup_2afc",
                        probe,
                        anchor,
                        meta={"target_shape": s_correct},
                    ),
                )

                self.stage.add_widget(row)

            Clock.schedule_once(show_choices, 2.0)


    def run_sample_2afc(self, item, probe: bool, anchor: bool):
        """Sample 2AFC with dementia-friendly card UI and big centered preview."""


        self.stage.clear_widgets()
        self.status_label.text = f"Trial {self.trial_idx}/{self.max_trials} — Sample"

        shapes = ["square", "circle", "triangle", "star", "diamond","rectangle","oval","pentagon","hexagon","cross","arrow"]

        s_correct, s_foil = random.sample(shapes, 2)
        seed = item.base_seed

        # Preview shape (large, centered)


        # Generate larger preview
        preview_arr = draw_shape(s_correct, size=360, seed=seed)
        self.show_instruction("Look carefully. This will disappear. Then tap the same one.")

        # Create the image
        preview_img = Image(
            texture=numpy_to_texture(preview_arr),
            size_hint=(None, None),
            height=dp(300),
            width=dp(300),
            allow_stretch=True,
            keep_ratio=True,
        )

        # Wrap inside a card
        preview_card = MDCard(
            orientation="vertical",
            size_hint=(None, None),
            width=dp(340),
            height=dp(340),
            elevation=8,
            shadow_softness=1,
            shadow_offset=(3, -3),
            md_bg_color=(0.98, 0.96, 0.9, 1),  # soft cream
            padding=20,
            radius=[20, 20, 20, 20],
        )
        preview_card.add_widget(preview_img)

        # Center the card on screen
        preview_box = AnchorLayout(
            anchor_x="center",
            anchor_y="center",
            size_hint=(1, None),
            height=dp(400),
        )
        preview_box.add_widget(preview_card)

        # Add to stage
        self.stage.add_widget(preview_box)


        def show_choices(_dt):
            if not self.session_active or self.is_paused:
                return
            self.stage.clear_widgets()
            self.show_instruction(f"Which one is the correct shape??")

            row = MDBoxLayout(
                orientation="vertical",
                size_hint=(1, None),
                height=dp(320),
                padding=20,
                spacing=20,
            )

            img_correct = draw_shape(s_correct, size=200, seed=seed)
            img_foil = draw_shape(s_foil, size=200, seed=seed + 5)
            choices = [img_correct, img_foil]
            labels = [s_correct, s_foil]

            if random.random() < 0.5:
                choices.reverse()
                labels.reverse()

            self._inject_choice_cards(
                row,
                choices,
                correct_idx=labels.index(s_correct),
                handler=lambda ok: self._handle_2afc_response(
                    ok,
                    item,
                    "sample_2afc",
                    probe,
                    anchor,
                    meta={"target_shape": s_correct},
                ),
            )

            self.stage.add_widget(row)

        Clock.schedule_once(show_choices, 2.0)

    def run_multi_afc(self, item, probe: bool, anchor: bool, n_choices: int = 3):
        """Intermediate stage: 3AFC or 4AFC before moving to grids."""

        self.stage.clear_widgets()
        self.status_label.text = f"Trial {self.trial_idx}/{self.max_trials} — {n_choices}AFC"

        shapes = ["square", "circle", "triangle", "star", "diamond",
                "rectangle", "oval", "pentagon", "hexagon", "cross", "arrow"]

        s_correct = random.choice(shapes)
        foils = random.sample([s for s in shapes if s != s_correct], n_choices - 1)
        seed = item.base_seed

        # --- Preview ---
        preview_arr = draw_shape(s_correct, size=300, seed=seed)
        self.show_instruction("Memorize this shape. Then pick the same one.")

        preview_img = Image(
            texture=numpy_to_texture(preview_arr),
            size_hint=(None, None),
            height=dp(280),
            width=dp(280),
        )
        preview_card = MDCard(size_hint=(None, None), width=dp(320), height=dp(320))
        preview_card.add_widget(preview_img)

        preview_box = AnchorLayout(anchor_x="center", anchor_y="center",
                                size_hint=(1, None), height=dp(360))
        preview_box.add_widget(preview_card)
        self.stage.add_widget(preview_box)

        # --- Show choices ---
        def show_choices(_dt):
            if not self.session_active or self.is_paused:
                return
            self.stage.clear_widgets()
            self.show_instruction(f"Which one is the correct shape?")

            row = MDBoxLayout(orientation="vertical", padding=20, spacing=20)

            # Build (image, label) pairs
            choices = [(draw_shape(s_correct, size=180, seed=seed), s_correct)]
            for i, f in enumerate(foils):
                choices.append((draw_shape(f, size=180, seed=seed + i * 5), f))

            # Shuffle pairs
            random.shuffle(choices)

            # Split back out into separate lists
            images = [img for img, lbl in choices]
            labels = [lbl for img, lbl in choices]

            # Find correct index by label
            correct_idx = labels.index(s_correct)

            self._inject_choice_cards(
                row,
                images,
                correct_idx=correct_idx,
                handler=lambda ok: self._handle_2afc_response(
                    ok,
                    item,
                    f"{n_choices}afc",
                    probe,
                    anchor,
                    meta={"target_shape": s_correct},
                ),
            )

            self.stage.add_widget(row)

        Clock.schedule_once(show_choices, 2.0)

    def _handle_2afc_response(self, correct: bool, item: Item, trial_type: str, probe: bool, anchor: bool, meta: Dict):
        rt = time.time() - self.trial_start_time if self.trial_start_time else None
        self.engine.update(item, int(correct))
        self.difficulty.push(int(correct))
        self.difficulty.tune()
        p_pred = AdaptiveEngine.prob_correct(self.engine.theta, item.a, item.b)
        rec = {
            "trial_index": self.trial_idx,
            "phase": self.phase,
            "trial_type": trial_type,
            "item_id": item.id,
            "item_family": item.family,
            "item_a": item.a,
            "item_b": item.b,
            "correct": bool(correct),
            "reaction_time_s": float(rt) if rt is not None else None,
            "theta_after": self.engine.theta,
            "predicted_p": float(p_pred),
            "probe": probe,
            "anchor": anchor,
            "difficulty_level": self.difficulty.level,
            "timestamp_end": datetime.utcnow().isoformat() + "Z",
        }
        rec.update(meta)
        self.trial_log.append(rec)

        self._feedback(correct)
        Clock.schedule_once(lambda dt: self.next_trial(), FEEDBACK_DELAY_S)
        
        

    def run_grid_find(self, item: Item, probe: bool, anchor: bool):
        self.stage.clear_widgets()
        self.status_label.text = f"Trial {self.trial_idx}/{self.max_trials} — Grid Find"

        # Difficulty-conditioned parameters
        rot_choices = self.difficulty.rotation_set()
        occ_lo, occ_hi = self.difficulty.occlusion_range()

        # Randomized trial condition
        trial_type = random.choice(["rotation", "occlusion", "distractor", "plain"])
        rotation_angle = random.choice(rot_choices) if trial_type == "rotation" else 0
        occlusion_frac = random.uniform(occ_lo, occ_hi) if trial_type == "occlusion" else 0.0

        seed = self.current_item.base_seed if self.current_item else random.randint(0, 1_000_000)

        # ----- Generate base sample (before transformations) -----
        # if item.family == "lsystem":
        #     base_sample = lsystem_image(size=360, seed=seed, iterations=4)
        if item.family == "lsystem":
            base_sample = lsystem_image(size=360, seed=seed, iterations=2 if self.difficulty.level < 4 else 4)
        elif item.family == "shape":
            base_sample = draw_shape(
                random.choice(["square", "circle", "triangle", "star", "heart", "diamond"]),
                size=360, seed=seed
            )
        else:
            fam = NoiseFamily[item.family.upper()]
            base_sample = fam.generate(size=360, seed=seed, difficulty=self.difficulty.level)

        # Apply trial transformation for preview
        preview_sample = base_sample.copy()
        if trial_type == "rotation":
            preview_sample = apply_rotation(preview_sample, rotation_angle)
        elif trial_type == "occlusion":
            preview_sample = apply_occlusion(preview_sample, occlusion_frac, seed=seed + 11)
        elif trial_type == "distractor":
            pil = PILImage.fromarray(preview_sample).convert("RGBA")
            d = ImageDraw.Draw(pil)
            w, h = pil.size
            d.ellipse((w*0.55, h*0.05, w*0.92, h*0.42), outline=(0,0,0,200), width=8)
            preview_sample = np.array(pil.convert("RGB"))

        self._preview_sample = preview_sample

        # Show preview image
        self.show_instruction("Memorize this picture. Then find it in the 3×3 grid.")
        sample_img = Image(texture=numpy_to_texture(preview_sample),
                        size_hint=(None, None), height=dp(280), width=dp(280))
        sample_box = AnchorLayout(anchor_x="center", anchor_y="center",
                                size_hint=(1, None), height=dp(340))
        sample_box.add_widget(sample_img)
        self.stage.add_widget(sample_box)

        def show_grid(_dt):
            if not self.session_active or self.is_paused:
                return
            self.stage.clear_widgets()

            container = MDBoxLayout(orientation="vertical", spacing=20, padding=20)
            
            instruction_card = MDCard(
                orientation="vertical",
                size_hint=(1, None),
                height=dp(80),
                padding=dp(6),
                radius=[12],
                md_bg_color=(1, 1, 1, 0.95),
                elevation=4,
            )

            instruction_lbl = Label(
                text="Find the matching picture and tap it.",
                size_hint=(1, None),
                height=dp(65),
                font_size="26sp",
                bold=True,
                halign="center",
                valign="middle",
                color=(0.1, 0.1, 0.1, 1),
            )
            instruction_lbl.bind(size=instruction_lbl.setter("text_size"))
            instruction_card.add_widget(instruction_lbl)

            container.add_widget(instruction_card)

            # compute tile size
            horiz_margin = dp(80)
            vert_margin = dp(240)
            available_w = max(200, Window.width - horiz_margin)
            available_h = max(200, Window.height - vert_margin)
            tile_px = int(min(available_w / 3.0 - dp(12),
                            available_h / 3.0 - dp(12),
                            IMAGE_TILE[0]))
            tile_px = int(max(MIN_TOUCH, tile_px))

            # grid setup
            grid = GridLayout(cols=3, spacing=20, padding=20, size_hint=(None, None))
            grid.width = int(tile_px * 3 + 20 * 2)
            grid.height = int(tile_px * 3 + 20 * 2)

            # center the grid
            grid_box = AnchorLayout(anchor_x="center", anchor_y="center",
                                    size_hint=(1, None), height=grid.height + dp(40))
            grid_box.add_widget(grid)
            container.add_widget(grid_box)
            self.stage.add_widget(container)

            # randomize target location
            target_pos = random.randint(0, 8)

            # Save trial metadata
            self._grid_meta = {
                "trial_index": self.trial_idx,
                "phase": self.phase,
                "trial_type": trial_type,
                "item_id": item.id,
                "item_family": item.family,
                "item_a": item.a,
                "item_b": item.b,
                "item_base_seed": item.base_seed,
                "rotation_angle": rotation_angle,
                "occlusion_frac": occlusion_frac,
                "target_pos": target_pos,
                "probe": probe,
                "anchor": anchor,
                "theta_before": self.engine.theta,
                "timestamp_start": datetime.utcnow().isoformat() + "Z",
                "difficulty_level": self.difficulty.level,
                "tile_px": tile_px
            }

            # Hard distractors if high difficulty
            hard_distractors = []
            if self.difficulty.level >= 5:
                hard_distractors = random.sample([p for p in range(9) if p != target_pos], k=3)

            seen_hashes = set()

            for pos in range(9):
                if pos == target_pos:
                    arr = self._preview_sample.copy()
                    arr = np.array(PILImage.fromarray(arr).resize((224, 224), PILImage.BILINEAR))
                else:
                    # Keep regenerating until unique
                    while True:
                        s = seed + 1000 + random.randint(1, 1_000_000)

                        if item.family == "lsystem":
                            arr = lsystem_image(size=224, seed=s, iterations=4)
                        elif item.family == "shape":
                            arr = draw_shape(
                                random.choice([
                                    "square", "circle", "triangle", "star", "diamond",
                                    "rectangle", "oval", "pentagon", "hexagon", "cross", "arrow"
                                ]),
                                size=224, seed=s
                            )
                        else:
                            fam = NoiseFamily[item.family.upper()]
                            arr = fam.generate(size=224, seed=s, difficulty=self.difficulty.level)

                        # trial-type transformations
                        if trial_type == "rotation":
                            if pos in hard_distractors:
                                jitter = random.choice([-30, -15, 15, 30])
                                arr = apply_rotation(arr, rotation_angle + jitter)
                            else:
                                arr = apply_rotation(arr, random.choice(rot_choices))

                        elif trial_type == "occlusion":
                            # Ensure multiple occluded distractors if preview is occluded
                            num_occluded_distractors = random.randint(2, 3)
                            occluded_positions = random.sample(
                                [p for p in range(9) if p != target_pos], num_occluded_distractors
                            )

                            for pos in range(9):
                                if pos == target_pos:
                                    arr = apply_occlusion(arr, occlusion_frac, seed=s + 7)
                                elif pos in occluded_positions:
                                    frac = occlusion_frac * random.uniform(0.8, 1.2)
                                    arr = apply_occlusion(arr, frac, seed=s + 9)
                                else:
                                    # normal distractor (unoccluded)
                                    pass


                        elif trial_type == "distractor":
                            pil = PILImage.fromarray(arr).convert("RGBA")
                            d = ImageDraw.Draw(pil)
                            w, h = pil.size
                            if pos in hard_distractors:
                                rx, ry = random.uniform(0.1, 0.7), random.uniform(0.1, 0.7)
                                d.ellipse((w*rx, h*ry, w*(rx+0.25), h*(ry+0.25)),
                                        outline=(0,0,0,200), width=6)
                            arr = np.array(pil.convert("RGB"))

                        # foil randomness
                        if random.random() < 0.10:
                            arr = np.fliplr(arr)
                        if random.random() < 0.10:
                            arr = np.rot90(arr, k=random.choice([0, 1, 3]))

                        # uniqueness check
                        h = hash_image(arr)
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            break  # unique image found

                # Convert to texture
                tex_pil = PILImage.fromarray(arr).resize((tile_px, tile_px), PILImage.BILINEAR)
                data = io.BytesIO()
                tex_pil.save(data, format="PNG")
                data.seek(0)
                core = CoreImage(data, ext="png")

                # Wrap in container with padding
                tile_container = BoxLayout(
                    orientation="vertical",
                    size_hint=(None, None),
                    width=tile_px,
                    height=tile_px,
                    padding=(0, 0, 0, dp(20))
                )

                tile_card = MDCard(
                    orientation="vertical",
                    size_hint=(None, None),
                    width=tile_px,
                    height=tile_px,
                    elevation=6,
                    shadow_softness=1,
                    shadow_offset=(2, -2),
                    md_bg_color=(0.98, 0.96, 0.9, 1),
                    radius=[12, 12, 12, 12],
                    ripple_behavior=True,
                )

                image_btn = ImageButton(texture=core.texture)
                image_btn.size_hint = (None, None)
                image_btn.width, image_btn.height = tile_px, tile_px
                image_btn.bind(on_press=lambda _b, p=pos: self._handle_grid_response(p))

                tile_card.add_widget(image_btn)
                tile_container.add_widget(tile_card)
                grid.add_widget(tile_container)
        Clock.schedule_once(show_grid, max(SAMPLE_PREVIEW_SEC, 2.8))


        
    def run_grid_find_2x2(self, item: Item, probe: bool, anchor: bool):
        """Intermediate stage: 2×2 grid before jumping to 3×3."""

        self.stage.clear_widgets()
        self.status_label.text = f"Trial {self.trial_idx}/{self.max_trials} — Grid 2×2"

        seed = item.base_seed

        # --- Generate base sample ---
        if item.family == "lsystem":
            base_sample = lsystem_image(size=280, seed=seed, iterations=2)  # easier
        elif item.family == "shape":
            base_sample = draw_shape(
                random.choice(["square", "circle", "triangle", "star", "diamond"]),
                size=280, seed=seed
            )
        else:
            fam = NoiseFamily[item.family.upper()]
            base_sample = fam.generate(size=280, seed=seed, difficulty=max(1, self.difficulty.level - 1))

        self._preview_sample = base_sample

        # --- Show preview ---
        self.show_instruction("Memorize this picture. Then find it in the 2×2 grid.")

        sample_img = Image(texture=numpy_to_texture(base_sample),
                        size_hint=(None, None), height=dp(240), width=dp(240))
        sample_box = AnchorLayout(anchor_x="center", anchor_y="center",
                                size_hint=(1, None), height=dp(300))
        sample_box.add_widget(sample_img)
        self.stage.add_widget(sample_box)

        def show_grid(_dt):
            if not self.session_active or self.is_paused:
                return
            self.stage.clear_widgets()

            grid = GridLayout(cols=2, spacing=20, padding=20,
                            size_hint=(None, None))
            tile_px = 200
            grid.width = tile_px * 2 + 40
            grid.height = tile_px * 2 + 40

            grid_box = AnchorLayout(anchor_x="center", anchor_y="center",
                                    size_hint=(1, None), height=grid.height + dp(40))
            grid_box.add_widget(grid)
            self.stage.add_widget(grid_box)

            target_pos = random.randint(0, 3)

            # --- Save trial metadata ---
            self._grid_meta = {
                "trial_index": self.trial_idx,
                "phase": self.phase,
                "trial_type": "2x2_grid",
                "item_id": item.id,
                "item_family": item.family,
                "item_a": item.a,
                "item_b": item.b,
                "item_base_seed": item.base_seed,
                "rotation_angle": 0,
                "occlusion_frac": 0.0,
                "target_pos": target_pos,
                "probe": probe,
                "anchor": anchor,
                "theta_before": self.engine.theta,
                "timestamp_start": datetime.utcnow().isoformat() + "Z",
                "difficulty_level": self.difficulty.level,
                "tile_px": tile_px,
            }

            # --- Populate grid ---
            for pos in range(4):
                if pos == target_pos:
                    arr = self._preview_sample.copy()
                else:
                    s = seed + 1000 + pos * 37
                    if item.family == "lsystem":
                        arr = lsystem_image(size=224, seed=s, iterations=2)
                    elif item.family == "shape":
                        arr = draw_shape(
                            random.choice(["square", "circle", "triangle", "star", "diamond"]),
                            size=224, seed=s
                        )
                    else:
                        fam = NoiseFamily[item.family.upper()]
                        arr = fam.generate(size=224, seed=s, difficulty=max(1, self.difficulty.level - 1))

                # Convert to texture
                tex_pil = PILImage.fromarray(arr).resize((tile_px, tile_px), PILImage.BILINEAR)
                data = io.BytesIO()
                tex_pil.save(data, format="PNG")
                data.seek(0)
                core = CoreImage(data, ext="png")

                image_btn = ImageButton(texture=core.texture)
                image_btn.size_hint = (None, None)
                image_btn.width, image_btn.height = tile_px, tile_px
                image_btn.bind(on_press=lambda _b, p=pos: self._handle_grid_response(p))

                tile_card = MDCard(size_hint=(None, None), width=tile_px, height=tile_px,
                                elevation=6, ripple_behavior=True)
                tile_card.add_widget(image_btn)
                grid.add_widget(tile_card)

        Clock.schedule_once(show_grid, 2.5)


    def _handle_grid_response(self, chosen_pos: int):
        """Handle grid response using self._grid_meta only."""
        meta = self._grid_meta
        target_pos = meta["target_pos"]
        item = Item(
            id=meta["item_id"],
            family=meta["item_family"],
            a=meta["item_a"],
            b=meta["item_b"],
            base_seed=meta["item_base_seed"]   # <-- use stored base_seed
        )       
        probe = meta["probe"]
        anchor = meta["anchor"]

        rt = time.time() - self.trial_start_time if self.trial_start_time else None
        correct = int(chosen_pos == target_pos)

        # Update model if not probe
        if not probe:
            self.engine.update(item, correct)
            self.difficulty.push(correct)
            self.difficulty.tune()

        # Save log
        p_pred = AdaptiveEngine.prob_correct(self.engine.theta, item.a, item.b)
        rec = dict(meta)
        rec.update({
            "chosen_pos": chosen_pos,
            "correct": bool(correct),
            "reaction_time_s": float(rt) if rt else None,
            "theta_after": self.engine.theta,
            "predicted_p": float(p_pred),
            "timestamp_end": datetime.utcnow().isoformat() + "Z",
            "probe_trial": probe,
            "anchor_trial": anchor,
        })
        self.trial_log.append(rec)

        # Feedback
        if not probe:
            self._feedback(bool(correct))

        if anchor:
            Logger.info(f"[ANCHOR] Trial {self.trial_idx} logged as anchor.")


        Clock.schedule_once(lambda dt: self.next_trial(), FEEDBACK_DELAY_S)



    # ===== summary helper =====
    def summary(self) -> Dict:
        if not self.trial_log:
            return {}
        df = pd.DataFrame(self.trial_log)
        acc = df.get('correct', pd.Series(dtype=float)).mean()
        rt = df.get('reaction_time_s', pd.Series(dtype=float)).mean()
        return {"n_trials": int(len(df)), "accuracy": float(acc), "avg_rt_s": float(rt), "difficulty": self.difficulty.level}
