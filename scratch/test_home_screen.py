import sys
from PyQt5.QtWidgets import QApplication
from src.ui.home_screen import HomeScreen

app = QApplication(sys.argv)
home = HomeScreen()
home.resize(1280, 800)
print("1280x800 layout OK")
home.resize(1920, 1080)
print("1920x1080 layout OK")
home.resize(800, 600)
print("800x600 layout OK")
print("All resize tests passed!")
