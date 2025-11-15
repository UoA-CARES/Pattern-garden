# ui/overhaul.py
"""
UI Overhaul for PatternGame (compact for iRobi screen)
- Professional, dementia-friendly interface
- Structured layout: header, instructions, controls, stage
- Compact large buttons (180x80) for small screens
- Choice cards auto-wrap in a 2-column grid
"""

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
from ui.game import PatternGame, WARM_BG
from config import style_button
from stimuli import numpy_to_texture, draw_shape
from kivy.uix.anchorlayout import AnchorLayout
from ui.game import Item
import random


class PatternGameUI(PatternGame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        Window.clearcolor = WARM_BG

        # Remove base UI we’re replacing
        self._remove_original_controls()
        if self._stage_instruction.parent:
            self._stage_instruction.parent.remove_widget(self._stage_instruction)

        # Build new layout
        self._build_header()
        self._build_instruction_band()
        self._build_controls()

    # -----------------------
    # UI Sections
    # -----------------------

    def _remove_original_controls(self):
        for child in list(self.children):
            if isinstance(child, MDBoxLayout) and any(
                isinstance(grand, MDRaisedButton) for grand in child.children
            ):
                self.remove_widget(child)

    def _build_header(self):
        header = MDBoxLayout(
            orientation="horizontal",
            size_hint=(1, None),
            height=dp(80),
            padding=10,
            spacing=10,
            md_bg_color=(0.96, 0.94, 0.88, 1),
        )

        try:
            logo = Image(
                source="assets/images/logo.png",
                size_hint=(None, None),
                height=dp(60),
                width=dp(120),
                allow_stretch=True,
                keep_ratio=True,
            )
            header.add_widget(logo)
        except Exception:
            header.add_widget(Label(size_hint=(None, 1), width=dp(120)))

        title = Label(
            text="Pattern Garden",
            font_size="28sp",
            bold=True,
            halign="center",
            valign="middle",
            color=(0.1, 0.1, 0.1, 1),
        )
        title.bind(size=title.setter("text_size"))
        header.add_widget(title)

        header.add_widget(Label(size_hint=(None, 1), width=dp(120)))
        self.add_widget(header, index=len(self.children))

    def _build_instruction_band(self):
        band = MDBoxLayout(
            orientation="vertical",
            size_hint=(1, None),
            height=dp(80),
            md_bg_color=(1, 0.97, 0.8, 1),
            padding=10,
        )

        self._stage_instruction.font_size = "26sp"
        self._stage_instruction.bold = True
        self._stage_instruction.halign = "center"
        self._stage_instruction.valign = "middle"
        self._stage_instruction.color = (0.05, 0.05, 0.05, 1)
        self._stage_instruction.bind(size=self._stage_instruction.setter("text_size"))

        band.add_widget(self._stage_instruction)
        self.add_widget(band, index=len(self.children))

    def _build_controls(self):
        self.controls = MDBoxLayout(
            size_hint=(1, None),
            height=dp(50),
            spacing=10,
            padding=10,
            md_bg_color=(0.96, 0.94, 0.88, 1),
        )

        def big_button(text, icon, on_press, color, disabled=False):
            btn = MDRaisedButton(
                text=text,
                icon=icon,
                font_size="20sp",
                size_hint=(None, None),
                width=dp(180),
                height=dp(80),
                on_release=on_press,
                disabled=disabled,
                md_bg_color=color,
                rounded_button=True,
                theme_text_color="Custom",
                text_color=(1, 1, 1, 1),
            )
            return btn

        self.start_btn = big_button("Start", "play", self.start_session, (0.2, 0.6, 0.3, 1))
        self.pause_btn = big_button("Break", "pause", self.toggle_pause, (0.2, 0.4, 0.7, 1), True)
        self.end_btn = big_button("Stop", "stop", self.end_session, (0.8, 0.2, 0.2, 1), True)
        self.save_btn = big_button("Save", "content-save", self.show_save_dialog, (0.5, 0.5, 0.5, 1), True)
        self.repeat_btn = big_button("Repeat", "repeat", lambda *_: self._repeat_instruction(), (0.1, 0.6, 0.6, 1), True)

        # 🔹 Trial counter label
        self.session_info = Label(
            text=f"Trial 0/{self.max_trials}",
            font_size="20sp",
            halign="center",
            valign="middle",
            color=(0.1, 0.1, 0.1, 1),
            size_hint=(None, None),
            width=dp(160),
        )
        self.session_info.bind(size=self.session_info.setter("text_size"))

        for b in [self.start_btn, self.pause_btn, self.end_btn, self.save_btn, self.repeat_btn]:
            self.controls.add_widget(b)

        # Add trial counter at the end of the control bar
        self.controls.add_widget(self.session_info)

        self.add_widget(self.controls, index=len(self.children))


    # -----------------------
    # Feedback & Dialogs
    # -----------------------

    def _feedback(self, correct: bool):
        if correct:
            msg = "Correct — well done."
            bg = (0.8, 1.0, 0.8, 1)
        else:
            msg = "Not quite — let's try the next one."
            bg = (1.0, 0.9, 0.7, 1)

        self.show_instruction(msg, speak=True)
        Window.clearcolor = bg
        Clock.schedule_once(lambda dt: setattr(Window, "clearcolor", WARM_BG), 1.5)

    def show_save_dialog(self, *args):
        if self.dialog:
            self.dialog.dismiss()

        self.participant_field = MDTextField(
            hint_text="Enter Participant ID (e.g., 001)",
            helper_text="Use de-identified code only",
            required=True,
            font_size="22sp",
            size_hint_x=1,
            input_filter="int",         # restricts to numbers only
            input_type="number"         # brings up numeric keypad on touch devices
        )

        self.dialog = MDDialog(
            title="Save Results",
            type="custom",
            content_cls=self.participant_field,
            buttons=[
                MDFlatButton(
                    text="CANCEL", font_size="20sp",
                    on_release=lambda x: self.dialog.dismiss()
                ),
                MDRaisedButton(
                    text="SAVE", font_size="20sp",
                    on_release=self.confirm_save
                ),
            ],
            auto_dismiss=False,
        )
        self.dialog.open()


    # -----------------------
    # Choice Cards
    # -----------------------

    def _make_choice_card(self, texture, label_text, on_press):
        card = MDCard(
            orientation="vertical",
            size_hint=(0.45, None),
            height=dp(260),
            elevation=6,
            shadow_softness=1,
            shadow_offset=(2, -2),
            md_bg_color=(0.98, 0.96, 0.9, 1),
            ripple_behavior=True,
        )

        shape_img = Image(texture=texture, size_hint=(1, 0.8))
        lbl = Label(
            text=label_text,
            font_size="20sp",
            halign="center",
            valign="middle",
            color=(0.1, 0.1, 0.1, 1),
            size_hint=(1, 0.2),
        )
        lbl.bind(size=lbl.setter("text_size"))

        card.add_widget(shape_img)
        card.add_widget(lbl)
        card.bind(on_release=on_press)
        return card

    def _inject_choice_cards(self, parent, choices, correct_idx, handler):
        grid = GridLayout(
            cols=2,
            spacing=15,
            padding=15,
            size_hint=(1, None),
            height=dp(280) * ((len(choices) + 1) // 2),
        )

        for i, arr in enumerate(choices):
            tex = numpy_to_texture(arr)
            card = self._make_choice_card(
                tex,
                f"Option {chr(65+i)}",
                lambda _b, ok=(i == correct_idx): handler(ok),
            )
            grid.add_widget(card)

        parent.add_widget(grid)

     