import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt5.QtWidgets import QApplication, QScrollArea
from src.ui.main_window import MainWindow
from src.ui.styles import APP_STYLESHEET

app = QApplication(sys.argv)
app.setStyleSheet(APP_STYLESHEET)

win = MainWindow()
win.resize(1440, 900)
win.show()
win._on_generate()
win.config_drawer.setGeometry(win.sim_page.width() - win.config_drawer.DRAWER_WIDTH, 0, win.config_drawer.DRAWER_WIDTH, win.sim_page.height())
win.config_drawer.open()
app.processEvents()

for c in win.config_drawer.findChildren(QScrollArea):
    c.verticalScrollBar().setValue(c.verticalScrollBar().maximum())

app.processEvents()
time.sleep(0.3)
app.processEvents()
win.grab().save("manual_assets/02b_config_drawer_disturbances.png")
print("Saved 02b_config_drawer_disturbances.png")
win.close()
app.quit()
