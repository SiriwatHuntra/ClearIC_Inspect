# STYLESHEET
STYLE = """
QMainWindow, QWidget#root {
    background: #5465FF;
}
QTabWidget::pane {
    background: #5465FF;
    border: none;
}
QTabBar::tab {
    background: #788BFF;
    color: #FFFFFF;
    padding: 6px 18px;
    border-radius: 4px 4px 0 0;
    font-size: 12px;
}
QTabBar::tab:selected {
    background: #5465FF;
    color: #FFFFFF;
    font-weight: bold;
}
QFrame#panel_right {
    background: #5465FF;
}
QFrame#setup_frame, QFrame#controls_frame {
    background: #788BFF;
    border-radius: 8px;
    padding: 8px;
}
QFrame#main_view {
    background: #788BFF;
    border-radius: 8px;
}
QFrame#image_area {
    background: #E2FDFF;
    border-radius: 8px;
}
QFrame#badge_area, QFrame#stats_area {
    background: #9BB1FF;
    border-radius: 8px;
    padding: 8px;
}
QFrame#badge_pass {
    background: #BFD7FF;
    border-radius: 8px;
    padding: 8px;
}
QFrame#badge_fail {
    background: #EF5350;
    border-radius: 8px;
    padding: 8px;
}
QFrame#badge_idle {
    background: #9BB1FF;
    border-radius: 8px;
    padding: 8px;
}
QFrame#error_banner {
    background: #EF5350;
    border-radius: 8px;
    padding: 6px;
}
QPushButton {
    background: #5465FF;
    color: #FFFFFF;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: bold;
    border: none;
}
QPushButton:disabled {
    background: #788BFF;
    color: #BFD7FF;
}
QLineEdit {
    background: #FFFFFF;
    color: #5465FF;
    border: 2px solid #5465FF;
    border-radius: 6px;
    padding: 4px 8px;
}
QLabel {
    color: #FFFFFF;
}
QLabel#stat_value {
    color: #E2FDFF;
    font-weight: bold;
}
QCheckBox {
    color: #FFFFFF;
    font-weight: bold;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #FFFFFF;
    border-radius: 3px;
    background: transparent;
}
QCheckBox::indicator:checked {
    background: #FFFFFF;
    image: none;
}
QCheckBox:disabled {
    color: #BFD7FF;
}
QCheckBox::indicator:disabled {
    border-color: #BFD7FF;
}
"""
