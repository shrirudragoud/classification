import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QLabel, QTextEdit,
                           QStackedWidget, QProgressBar, QMessageBox)
from PyQt6.QtGui import QPalette, QColor, QPixmap, QFont
from PyQt6.QtCore import Qt
from data_loader import download_dataset, load_dataset
from model import create_model, load_model_from_checkpoint
from testing_tab import TestingTab
from training_tab import TrainingTab
from config import *

class ConsoleOutput(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #d4d4d4;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                padding: 10px;
                border: 1px solid #2d2d2d;
                border-radius: 5px;
            }
        """)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Hex Acadmey")
        self.setGeometry(100, 100, 1400, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Create banner
        banner = QWidget()
        banner.setFixedHeight(60)
        banner.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                border-bottom: 1px solid #2d2d2d;
            }
        """)
        banner_layout = QHBoxLayout()
        banner_layout.setContentsMargins(20, 0, 20, 0)
        
        # Logo
        logo_label = QLabel()
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public", "favicon.ico")
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            logo_label.setPixmap(logo_pixmap.scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio))
        else:
            logo_label.setText("AI")
            logo_label.setStyleSheet("""
                QLabel {
                    color: #d4d4d4;
                    font-size: 24px;
                    font-weight: bold;
                    font-family: 'Consolas', monospace;
                }
            """)
        
        # Title
        title = QLabel("Hex Academy")
        title.setStyleSheet("""
            QLabel {
                color: #d4d4d4;
                font-size: 24px;
                font-weight: bold;
                font-family: 'Consolas', monospace;
            }
        """)
        
        # Subtitle
        subtitle = QLabel("Supervised Learning Module")
        subtitle.setStyleSheet("""
            QLabel {
                color: #808080;
                font-size: 12px;
                font-family: 'Consolas', monospace;
            }
        """)
        
        # Add banner components
        title_layout = QVBoxLayout()
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        
        banner_layout.addWidget(logo_label)
        banner_layout.addSpacing(10)
        banner_layout.addLayout(title_layout)
        banner_layout.addStretch()
        banner.setLayout(banner_layout)
        
        # Add banner to main layout
        main_layout.addWidget(banner)
        
        # Create content area
        content = QHBoxLayout()
        
        # Create sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                border-right: 1px solid #2d2d2d;
            }
            QPushButton {
                text-align: left;
                padding: 15px;
                border: none;
                border-radius: 0;
                margin: 0;
                color: #d4d4d4;
                font-family: 'Consolas', monospace;
                font-size: 14px;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: #2a2a2a;
            }
            QPushButton:checked {
                background-color: #2d2d2d;
                border-left: 3px solid #007acc;
            }
        """)
        
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setSpacing(0)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create sidebar buttons
        self.data_btn = QPushButton("Data Management")
        self.data_btn.setCheckable(True)
        self.train_btn = QPushButton("Model Training")
        self.train_btn.setCheckable(True)
        self.eval_btn = QPushButton("Model Evaluation")
        self.eval_btn.setCheckable(True)
        self.test_btn = QPushButton("Live Testing")
        self.test_btn.setCheckable(True)
        
        # Connect buttons
        self.data_btn.clicked.connect(lambda: self.switch_page(0))
        self.train_btn.clicked.connect(lambda: self.switch_page(1))
        self.eval_btn.clicked.connect(lambda: self.switch_page(2))
        self.test_btn.clicked.connect(lambda: self.switch_page(3))
        
        # Add buttons to sidebar
        sidebar_layout.addWidget(self.data_btn)
        sidebar_layout.addWidget(self.train_btn)
        sidebar_layout.addWidget(self.eval_btn)
        sidebar_layout.addWidget(self.test_btn)
        sidebar_layout.addStretch()
        sidebar.setLayout(sidebar_layout)
        
        # Create stacked widget for different pages
        self.stack = QStackedWidget()
        
        # Create pages
        data_page = self.create_data_page()
        train_page = TrainingTab()
        eval_page = self.create_eval_page()
        test_page = TestingTab()
        
        self.stack.addWidget(data_page)
        self.stack.addWidget(train_page)
        self.stack.addWidget(eval_page)
        self.stack.addWidget(test_page)
        
        # Add widgets to content layout
        content.addWidget(sidebar)
        content.addWidget(self.stack)
        
        # Add content to main layout
        main_layout.addLayout(content)
        
        # Set default page
        self.data_btn.setChecked(True)
        
        # Set application style
        self.set_style()
    
    def switch_page(self, index):
        # Uncheck all buttons
        for btn in [self.data_btn, self.train_btn, self.eval_btn, self.test_btn]:
            btn.setChecked(False)
        
        # Check the clicked button
        [self.data_btn, self.train_btn, self.eval_btn, self.test_btn][index].setChecked(True)
        
        # Switch page
        self.stack.setCurrentIndex(index)
    
    def create_data_page(self):
        page = QWidget()
        layout = QHBoxLayout()
        
        # Left side - Controls and Terminal
        left_side = QWidget()
        left_layout = QVBoxLayout()
        
        # Control buttons
        controls = QWidget()
        controls_layout = QVBoxLayout()
        
        button_style = """
            QPushButton {
                background-color: #2d2d2d;
                color: #d4d4d4;
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Consolas', monospace;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
            QPushButton:pressed {
                background-color: #4d4d4d;
            }
        """
        
        self.download_btn = QPushButton("Download Dataset")
        self.download_btn.setStyleSheet(button_style)
        self.download_btn.clicked.connect(self.download_data)
        
        self.preprocess_btn = QPushButton("Preprocess Data")
        self.preprocess_btn.setStyleSheet(button_style)
        self.preprocess_btn.clicked.connect(self.preprocess_data)
        
        controls_layout.addWidget(self.download_btn)
        controls_layout.addWidget(self.preprocess_btn)
        controls.setLayout(controls_layout)
        
        # Terminal output
        self.data_console = ConsoleOutput()
        
        left_layout.addWidget(controls)
        left_layout.addWidget(QLabel("System Output"))
        left_layout.addWidget(self.data_console)
        left_side.setLayout(left_layout)
        
        layout.addWidget(left_side)
        page.setLayout(layout)
        return page
    
    def create_eval_page(self):
        page = QWidget()
        layout = QHBoxLayout()
        
        # Left side - Controls and Terminal
        left_side = QWidget()
        left_layout = QVBoxLayout()
        
        # Control buttons
        button_style = """
            QPushButton {
                background-color: #2d2d2d;
                color: #d4d4d4;
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Consolas', monospace;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
            QPushButton:pressed {
                background-color: #4d4d4d;
            }
        """
        
        self.eval_btn = QPushButton("Evaluate Model")
        self.eval_btn.setStyleSheet(button_style)
        self.eval_btn.clicked.connect(self.evaluate_model)
        
        # Terminal output
        self.eval_console = ConsoleOutput()
        
        left_layout.addWidget(self.eval_btn)
        left_layout.addWidget(QLabel("System Output"))
        left_layout.addWidget(self.eval_console)
        left_side.setLayout(left_layout)
        
        layout.addWidget(left_side)
        page.setLayout(layout)
        return page
    
    def download_data(self):
        self.download_btn.setEnabled(False)
        self.data_console.append("Initiating dataset download...")
        try:
            download_dataset()
            self.data_console.append("Dataset transfer complete.")
            self.data_console.append("Status: SUCCESS")
        except Exception as e:
            self.data_console.append("Error encountered during transfer.")
            self.data_console.append(f"Error details: {str(e)}")
            self.data_console.append("Status: FAILED")
        finally:
            self.download_btn.setEnabled(True)
    
    def preprocess_data(self):
        self.preprocess_btn.setEnabled(False)
        self.data_console.append("Initializing data preprocessing...")
        try:
            x_train, y_train, _ = load_dataset(TRAIN_DIR)
            self.data_console.append(f"Processing complete: {len(x_train)} samples")
            self.data_console.append("Data normalization complete")
            self.data_console.append("Status: SUCCESS")
        except Exception as e:
            self.data_console.append("Error in preprocessing pipeline")
            self.data_console.append(f"Error details: {str(e)}")
            self.data_console.append("Status: FAILED")
        finally:
            self.preprocess_btn.setEnabled(True)
    
    def evaluate_model(self):
        self.eval_btn.setEnabled(False)
        self.eval_console.append("Initiating model evaluation...")
        try:
            model = load_model_from_checkpoint()
            if model is None:
                raise Exception("Model not found in specified location")
                
            self.eval_console.append("Loading test dataset...")
            x_train, y_train, _ = load_dataset(TRAIN_DIR)
            
            self.eval_console.append("Running evaluation protocol...")
            loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
            
            self.eval_console.append("\nEvaluation Results:")
            self.eval_console.append(f"Model Loss: {loss:.4f}")
            self.eval_console.append(f"Model Accuracy: {accuracy:.4f}")
            self.eval_console.append("Status: SUCCESS")
        except Exception as e:
            self.eval_console.append("Error in evaluation protocol")
            self.eval_console.append(f"Error details: {str(e)}")
            self.eval_console.append("Status: FAILED")
        finally:
            self.eval_btn.setEnabled(True)
    
    def set_style(self):
        QApplication.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(26, 26, 26))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(212, 212, 212))
        palette.setColor(QPalette.ColorRole.Base, QColor(18, 18, 18))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(45, 45, 45))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(212, 212, 212))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(212, 212, 212))
        palette.setColor(QPalette.ColorRole.Text, QColor(212, 212, 212))
        palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(212, 212, 212))
        palette.setColor(QPalette.ColorRole.Link, QColor(0, 122, 204))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 122, 204))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        QApplication.setPalette(palette)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()