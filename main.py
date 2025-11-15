
from kivymd.app import MDApp
from ui.overhaul import PatternGameUI as PatternGame

class PatternGardenApp(MDApp):
    def build(self):
        return PatternGame()


if __name__ == "__main__":
    PatternGardenApp().run()
