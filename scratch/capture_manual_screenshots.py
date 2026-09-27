"""
Script to capture pristine screenshots of SADHA for the User Manual.
"""
import sys
import os
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap

# Ensure outputs directory exists
os.makedirs("manual_assets", exist_ok=True)

from src.ui.main_window import MainWindow
from src.ui.styles import APP_STYLESHEET
from src.core.config import apply_preset, SimulationConfig

def capture():
    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("SADHA")
    app.setStyleSheet(APP_STYLESHEET)

    win = MainWindow()
    win.resize(1440, 900)
    win.show()
    app.processEvents()

    # 1. Home Screen
    win._stack.setCurrentIndex(0)
    app.processEvents()
    time.sleep(0.3)
    app.processEvents()
    win.grab().save("manual_assets/01_home_screen.png")
    print("Saved: 01_home_screen.png")

    # 2. Configuration Overlay open over simulation page
    win._on_generate()
    app.processEvents()
    win.config_drawer.open()
    app.processEvents()
    time.sleep(0.3)
    app.processEvents()
    win.grab().save("manual_assets/02_config_drawer.png")
    print("Saved: 02_config_drawer.png")

    # 3. Running Simulation (Nominal / High-Dynamics Fig8)
    win.config_drawer.close_drawer()
    win._stop_simulation()
    cfg = apply_preset("HIGH_DYNAMICS_FIG8")
    win._init_simulation(cfg)
    win._start_simulation()
    
    # Run 60 ticks to get stable tracking, trails, and sparklines
    for _ in range(60):
        win._tick()
        app.processEvents()
        time.sleep(0.016)

    app.processEvents()
    win.grab().save("manual_assets/03_simulation_running.png")
    print("Saved: 03_simulation_running.png")

    # 4. Severe Fog & Noise Disturbance
    win._stop_simulation()
    cfg_fog = apply_preset("SEVERE_FOG_NOISE")
    win._init_simulation(cfg_fog)
    win._start_simulation()
    for _ in range(50):
        win._tick()
        app.processEvents()
        time.sleep(0.016)

    app.processEvents()
    win.grab().save("manual_assets/04_disturbances_fog.png")
    print("Saved: 04_disturbances_fog.png")

    # 5. Multi-Beacon Crossing Scenario
    win._stop_simulation()
    cfg_multi = apply_preset("MULTI_BEACON_CROSSING")
    win._init_simulation(cfg_multi)
    win._start_simulation()
    for _ in range(60):
        win._tick()
        app.processEvents()
        time.sleep(0.016)

    app.processEvents()
    win.grab().save("manual_assets/05_multi_beacon_tracking.png")
    print("Saved: 05_multi_beacon_tracking.png")

    # 6. Video Mode — Raw Feed
    win._stop_simulation()
    video_path = os.path.abspath("spot_jitter.mp4")
    if os.path.exists(video_path):
        win._on_upload_video(video_path)
        # Advance 20 frames
        for _ in range(25):
            win._video_tick()
            app.processEvents()
            time.sleep(0.02)
        app.processEvents()
        win.grab().save("manual_assets/06_video_mode_raw.png")
        print("Saved: 06_video_mode_raw.png")

        # 7. Video Mode — Trajectory HUD
        win._set_left_video_view("traj")
        for _ in range(20):
            win._video_tick()
            app.processEvents()
            time.sleep(0.02)
        app.processEvents()
        win.grab().save("manual_assets/07_video_mode_trajectory.png")
        print("Saved: 07_video_mode_trajectory.png")

    # 8. Export Log triggering (mock QMessageBox to avoid modal blocking)
    from PyQt5.QtWidgets import QMessageBox
    QMessageBox.information = lambda *args, **kwargs: None
    win._export_log()
    app.processEvents()
    print("Export log completed!")

    win.close()
    app.quit()
    print("All captures completed successfully!")

if __name__ == "__main__":
    capture()
