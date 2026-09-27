"""
SADHA — UI Visual Design System
Monochromatic high-contrast industrial grayscale palette:
  • 131313 (Night)
  • 3C3C3C (Onyx)
  • 646464 (Dim gray)
  • B5B5B5 (Silver)
  • F3F3F3 (White smoke)
"""

# ---------------------------------------------------------------------------
# 5-Color Monochromatic Grayscale Palette
# ---------------------------------------------------------------------------
PALETTE = {
    # Palette definition from swatch image:
    "night":          "#131313",   # Night — deep obsidian black
    "onyx":           "#3C3C3C",   # Onyx — dark charcoal
    "dim_gray":       "#646464",   # Dim gray — medium neutral gray
    "silver":         "#B5B5B5",   # Silver — light silver border/divider
    "white_smoke":    "#F3F3F3",   # White smoke — crisp canvas background

    # Functional role assignments:
    "primary":        "#131313",   # Night — top chrome, primary action buttons
    "secondary":      "#3C3C3C",   # Onyx — secondary accents & controls
    "neutral":        "#F3F3F3",   # White smoke — surface backgrounds
    "background":     "#F3F3F3",   # White smoke — page background
    "card_bg":        "#FFFFFF",   # Pure white cards on white smoke canvas
    "border":         "#B5B5B5",   # Silver — dividers, input outlines, borders
    "grid_light":     "#B5B5B5",   # Silver — light surface grid / dividers
    "text_primary":   "#131313",   # Night — crisp high-contrast primary text
    "text_secondary": "#3C3C3C",   # Onyx — body text and descriptions
    "text_muted":     "#646464",   # Dim gray — captions & labels
    
    # Viewport dark canvas (World view & Camera sensor view)
    "dark_surface":   "#131313",   # Night — deep canvas background
    "dark_grid":      "#3C3C3C",   # Onyx — grid lines in dark views
    "dark_accent":    "#646464",   # Dim gray — center crosshairs
    "dark_text":      "#F3F3F3",   # White smoke — HUD overlay telemetry text
    "beacon_glow":    "#B5B5B5",   # Silver beacon glow
    "frustum_fill":   "rgba(181, 181, 181, 35)", # Silver FOV tint
    "frustum_edge":   "#B5B5B5",   # Silver FOV border

    # Mission-critical telemetry states
    "locked":         "#2E7D32",   # Telemetry green — locked state
    "lost":           "#C62828",   # Telemetry red — lost state
    "searching":      "#EF6C00",   # Telemetry amber — searching indicator
}

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
FONT_MONO    = "'Cascadia Mono', Consolas, 'Courier New', monospace"
FONT_HEADING = "'Segoe UI', 'Inter', Arial, Helvetica, sans-serif"
FONT_BODY    = "'Segoe UI', 'Inter', Arial, Helvetica, sans-serif"

# ---------------------------------------------------------------------------
# Qt Stylesheet — global application skin
# ---------------------------------------------------------------------------
APP_STYLESHEET = f"""
QWidget {{
    background-color: {PALETTE['background']};
    color: {PALETTE['text_primary']};
    font-family: {FONT_HEADING};
    font-size: 14px;
}}

/* ---- Top Header Bar ---- */
#header_bar {{
    background-color: {PALETTE['primary']};
    border-bottom: 2px solid {PALETTE['secondary']};
    min-height: 48px;
    max-height: 48px;
}}
#header_bar QLabel {{
    color: #FFFFFF;
    background: transparent;
}}
#header_bar QWidget {{
    background-color: transparent;
}}
#sim_controls_widget, #video_controls_widget {{
    background-color: transparent;
    background: transparent;
}}
#sadha_wordmark {{
    font-family: {FONT_HEADING};
    font-size: 20px;
    font-weight: 800;
    letter-spacing: 3px;
    color: #FFFFFF;
    padding-left: 16px;
}}
#start_stop_btn {{
    background-color: {PALETTE['locked']};
    color: #FFFFFF;
    border: 1px solid #4CAF50;
    border-radius: 4px;
    padding: 6px 20px;
    font-family: {FONT_HEADING};
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 0.5px;
    margin-right: 14px;
}}
#start_stop_btn:hover {{
    background-color: #388E3C;
    border-color: #81C784;
}}
#start_stop_btn[running="true"] {{
    background-color: {PALETTE['lost']};
    color: #FFFFFF;
    border: 1px solid #EF5350;
}}
#start_stop_btn[running="true"]:hover {{
    background-color: #D32F2F;
    border-color: #E57373;
}}

/* ---- Panel frames ---- */
#world_panel, #camera_panel {{
    background-color: {PALETTE['dark_surface']};
    border: 1px solid {PALETTE['dark_grid']};
    border-radius: 4px;
}}
#panel_label {{
    font-family: {FONT_HEADING};
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    color: {PALETTE['text_primary']};
    padding: 6px 10px;
    background: transparent;
}}

/* ---- Analytics Dock ---- */
#analytics_dock {{
    background-color: #FFFFFF;
    border-top: 1px solid {PALETTE['border']};
    min-height: 140px;
}}
.metric_value {{
    font-family: {FONT_MONO};
    font-size: 22px;
    font-weight: bold;
    color: {PALETTE['text_primary']};
}}
.metric_label {{
    font-family: {FONT_HEADING};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    color: {PALETTE['text_muted']};
    text-transform: uppercase;
}}
.metric_unit {{
    font-family: {FONT_HEADING};
    font-size: 12px;
    font-weight: 600;
    color: {PALETTE['text_secondary']};
}}

/* ---- Configuration Drawer ---- */
#config_overlay {{
    background-color: {PALETTE['background']};
    border-left: 2px solid {PALETTE['secondary']};
}}
#config_overlay QLabel {{
    background: transparent;
    color: {PALETTE['text_primary']};
    font-family: {FONT_BODY};
    font-size: 13px;
    font-weight: normal;
}}
#config_section_title {{
    font-family: {FONT_HEADING};
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 2px;
    color: {PALETTE['text_primary']};
    padding: 10px 0 4px 0;
    border-bottom: 2px solid {PALETTE['border']};
}}
#config_overlay QSpinBox, #config_overlay QDoubleSpinBox, #config_overlay QComboBox {{
    background-color: #FFFFFF;
    color: {PALETTE['text_primary']};
    border: 1px solid {PALETTE['border']};
    border-radius: 3px;
    padding: 3px 6px;
    font-family: {FONT_BODY};
    font-size: 13px;
    font-weight: normal;
    min-height: 26px;
}}
#config_overlay QCheckBox {{
    font-family: {FONT_BODY};
    font-size: 13px;
    font-weight: normal;
    color: {PALETTE['text_primary']};
    spacing: 6px;
}}
#config_overlay QPushButton {{
    background-color: {PALETTE['primary']};
    color: #FFFFFF;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-family: {FONT_HEADING};
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 0.5px;
}}
#config_overlay QPushButton:hover {{
    background-color: {PALETTE['secondary']};
}}

/* ---- Home Screen ---- */
#home_screen {{
    background-color: {PALETTE['background']};
}}
.home_option_card {{
    background-color: #FFFFFF;
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    padding: 24px;
}}
.home_option_card:hover {{
    border: 2px solid {PALETTE['primary']};
}}

/* ---- Lock status badge ---- */
#lock_badge[state="locked"] {{
    background-color: {PALETTE['locked']};
    color: #FFFFFF;
    border-radius: 3px;
    padding: 2px 10px;
    font-family: {FONT_HEADING};
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 0.5px;
}}
#lock_badge[state="lost"] {{
    background-color: {PALETTE['lost']};
    color: #FFFFFF;
    border-radius: 3px;
    padding: 2px 10px;
    font-family: {FONT_HEADING};
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 0.5px;
}}

/* ---- Scrollbars ---- */
QScrollBar:vertical {{
    background: {PALETTE['neutral']};
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {PALETTE['border']};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {PALETTE['secondary']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* ---- Video Scrubber Slider ---- */
QSlider::groove:horizontal {{
    border: 1px solid #3C3C3C;
    height: 6px;
    background: #2A2A2A;
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: #2E7D32;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: #FFFFFF;
    border: 1px solid #131313;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background: #B5B5B5;
}}
"""
