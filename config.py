from kivy.metrics import dp
from kivy.uix.button import Button

# Colors / theme
BG = (0.97, 0.95, 0.88, 1)  # warm beige background
PANEL_BG = (1, 1, 1, 1)
BTN_BG = (0.08, 0.44, 0.32, 1)  # deep green
BTN_TEXT = (1, 1, 1, 1)
ALT_BTN_BG = (0.85, 0.2, 0.2, 1)  # red for stop
TEXT_COLOR = (0.07, 0.09, 0.11, 1)

# Typography / sizing
LARGE_FONT = '28sp'
MED_FONT = '20sp'
SMALL_FONT = '16sp'
BUTTON_HEIGHT = dp(72)
IMAGE_TILE = (224, 224)

# Timings (s)
SAMPLE_PREVIEW_SEC = 1.4
WARMUP_PREVIEW_SEC = 1.4


def style_button(btn: Button, primary: bool = True):
    btn.size_hint_y = None
    btn.height = BUTTON_HEIGHT
    btn.font_size = LARGE_FONT
    btn.color = BTN_TEXT
    btn.background_normal = ''
    btn.background_color = BTN_BG if primary else (0.6, 0.6, 0.63, 1)
