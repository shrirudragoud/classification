import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QProgressBar, QFileDialog, QGroupBox,
                           QTextEdit)
from PyQt6.QtCore import Qt, QMimeData
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent
import cv2
import numpy as np
from PIL import Image
from model import load_model_from_checkpoint
from config import *

class ImageDropWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Drop sample image or click to browse")
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #3d3d3d;
                border-radius: 5px;
                padding: 20px;
                background: #1a1a1a;
                color: #d4d4d4;
                font-family: 'Consolas', monospace;
            }
        """)
        self.setMinimumSize(300, 300)
        self.setAcceptDrops(True)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasImage() or event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasImage():
            self.handle_image(event.mimeData().imageData())
        elif event.mimeData().hasUrls():
            url = event.mimeData().urls()[0].toLocalFile()
            if url.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.load_image(url)
                
    def handle_image(self, image_data):
        pixmap = QPixmap.fromImage(image_data)
        self.set_image(pixmap)
        
    def load_image(self, path):
        pixmap = QPixmap(path)
        self.set_image(pixmap)
        if hasattr(self, 'image_loaded_callback'):
            self.image_loaded_callback(path)
        
    def set_image(self, pixmap):
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)

class TestingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.current_image_path = None
        self.init_ui()
        self.load_model()
        
    def init_ui(self):
        layout = QHBoxLayout()
        
        # Style definitions
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
            QPushButton:disabled {
                background-color: #1d1d1d;
                color: #808080;
                border-color: #2d2d2d;
            }
        """
        
        progress_style = """
            QProgressBar {
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                text-align: center;
                background-color: #1a1a1a;
                color: #d4d4d4;
                font-family: 'Consolas', monospace;
            }
            QProgressBar::chunk {
                background-color: #007acc;
            }
        """
        
        # Left side - Controls and Terminal
        left_side = QWidget()
        left_layout = QVBoxLayout()
        
        # Controls
        self.browse_button = QPushButton("Load Sample")
        self.browse_button.setStyleSheet(button_style)
        self.browse_button.clicked.connect(self.browse_image)
        
        self.batch_button = QPushButton("Batch Analysis")
        self.batch_button.setStyleSheet(button_style)
        self.batch_button.clicked.connect(self.process_batch)
        
        self.batch_progress = QProgressBar()
        self.batch_progress.setStyleSheet(progress_style)
        self.batch_progress.setVisible(False)
        
        # Terminal Output
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #d4d4d4;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                padding: 10px;
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                line-height: 1.2;
            }
        """)
        
        left_layout.addWidget(self.browse_button)
        left_layout.addWidget(self.batch_button)
        left_layout.addWidget(self.batch_progress)
        left_layout.addWidget(QLabel("System Output:"))
        left_layout.addWidget(self.console)
        left_side.setLayout(left_layout)
        
        # Right side - Image and Results
        right_side = QWidget()
        right_layout = QVBoxLayout()
        
        # Image preview
        image_group = QGroupBox("Sample Input")
        image_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                margin-top: 12px;
                font-family: 'Consolas', monospace;
                color: #d4d4d4;
            }
            QGroupBox::title {
                color: #d4d4d4;
            }
        """)
        image_layout = QVBoxLayout()
        self.image_widget = ImageDropWidget()
        self.image_widget.image_loaded_callback = self.process_image
        image_layout.addWidget(self.image_widget)
        image_group.setLayout(image_layout)
        
        # Prediction results
        results_group = QGroupBox("Analysis Results")
        results_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                margin-top: 12px;
                font-family: 'Consolas', monospace;
                color: #d4d4d4;
            }
            QGroupBox::title {
                color: #d4d4d4;
            }
            QLabel {
                color: #d4d4d4;
                font-family: 'Consolas', monospace;
            }
        """)
        results_layout = QVBoxLayout()
        
        self.result_label = QLabel("Awaiting sample image...")
        self.result_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #d4d4d4;
        """)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setStyleSheet(progress_style)
        
        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        results_layout.addWidget(self.result_label)
        results_layout.addWidget(self.confidence_bar)
        results_layout.addWidget(self.confidence_label)
        results_group.setLayout(results_layout)
        
        right_layout.addWidget(image_group)
        right_layout.addWidget(results_group)
        right_side.setLayout(right_layout)
        
        # Add both sides to main layout
        layout.addWidget(left_side)
        layout.addWidget(right_side)
        
        self.setLayout(layout)
    
    def load_model(self):
        try:
            self.console.append("Loading model...")
            self.model = load_model_from_checkpoint()
            if self.model:
                self.console.append("Model loaded successfully.")
        except Exception as e:
            self.console.append(f"Error loading model: {str(e)}")
    
    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.image_widget.load_image(file_path)
    
    def process_image(self, image_path):
        if not self.model:
            self.result_label.setText("Error: Model not loaded")
            self.console.append("Error: Model not initialized.")
            return
            
        try:
            self.console.append(f"\nProcessing image: {os.path.basename(image_path)}")
            
            # Load and preprocess image
            self.console.append("Loading image...")
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Failed to load image")
            
            self.console.append("Preprocessing image...")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            self.console.append(f"Image resized to {IMG_SIZE}x{IMG_SIZE}")
            
            img = np.expand_dims(img, axis=-1)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Make prediction
            self.console.append("Running analysis...")
            prediction = self.model.predict(img, verbose=0)[0][0]
            
            # Update UI
            confidence = prediction if prediction >= 0.5 else 1 - prediction
            confidence_pct = confidence * 100
            
            result = "DOG" if prediction >= 0.5 else "CAT"
            self.result_label.setText(f"Result: {result}")
            
            self.confidence_bar.setValue(int(confidence_pct))
            self.confidence_label.setText(f"Confidence: {confidence_pct:.1f}%")
            
            self.console.append(f"Analysis complete: {result}")
            self.console.append(f"Confidence level: {confidence_pct:.1f}%")
            
        except Exception as e:
            self.console.append(f"Error processing image: {str(e)}")
            self.result_label.setText("Analysis Error")
            self.confidence_bar.setValue(0)
            self.confidence_label.setText("Confidence: -")
    
    def process_batch(self):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with Images"
        )
        if not folder_path:
            return
            
        try:
            image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            
            if not image_files:
                self.console.append("No images found in selected folder.")
                return
                
            self.batch_progress.setVisible(True)
            self.batch_progress.setMaximum(len(image_files))
            self.batch_progress.setValue(0)
            
            self.console.append(f"\nBatch analysis initiated:")
            self.console.append(f"Processing {len(image_files)} images...")
            
            results = []
            for i, img_file in enumerate(image_files):
                img_path = os.path.join(folder_path, img_file)
                try:
                    self.console.append(f"\nProcessing: {img_file}")
                    
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = np.expand_dims(img, axis=-1)
                    img = img.astype(np.float32) / 255.0
                    img = np.expand_dims(img, axis=0)
                    
                    # Predict
                    prediction = self.model.predict(img, verbose=0)[0][0]
                    confidence = prediction if prediction >= 0.5 else 1 - prediction
                    result = "DOG" if prediction >= 0.5 else "CAT"
                    
                    results.append({
                        'file': img_file,
                        'prediction': result,
                        'confidence': confidence * 100
                    })
                    
                    self.console.append(
                        f"Result: {result} ({confidence * 100:.1f}% confidence)"
                    )
                    
                except Exception as e:
                    self.console.append(f"Error processing {img_file}: {str(e)}")
                    results.append({
                        'file': img_file,
                        'prediction': 'ERROR',
                        'confidence': 0
                    })
                
                self.batch_progress.setValue(i + 1)
            
            # Save results to CSV
            import pandas as pd
            df = pd.DataFrame(results)
            output_path = os.path.join(folder_path, 'analysis_results.csv')
            df.to_csv(output_path, index=False)
            
            self.console.append("\nBatch analysis complete.")
            self.console.append(f"Results exported to: {output_path}")
            
        except Exception as e:
            self.console.append(f"\nBatch analysis error: {str(e)}")
        finally:
            self.batch_progress.setVisible(False)