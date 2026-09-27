"""
SADHA — Primary Application Entry Point
AI-Based Virtual Camera Tracking System for Coarse Alignment of Mobile FSOC Terminals
Smart India Hackathon (SIH) — Problem Statement 4

Initializes the high-DPI PyQt5 application lifecycle, applies global dark-mode
styling tokens, constructs the primary MainWindow orchestrator, and starts the
Qt event loop.

Usage:
    python main.py
"""

import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from src.ui.main_window import MainWindow
from src.ui.styles import APP_STYLESHEET


def main():
    """Configure Qt application attributes and launch the SADHA GUI."""
    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("SADHA")
    app.setOrganizationName("SADHA-FSOC")
    app.setStyleSheet(APP_STYLESHEET)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
