import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QProgressBar, QTextEdit, QSpinBox, 
                           QDoubleSpinBox, QGroupBox, QGridLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from config import *
from model import create_model, save_model
from data_loader import load_dataset, create_data_generators

class TrainingThread(QThread):
    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    log = pyqtSignal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def run(self):
        try:
            self.log.emit("Initializing training sequence...")
            self.log.emit("Loading dataset...")
            x_train, y_train, _ = load_dataset(TRAIN_DIR)
            self.log.emit(f"{len(x_train)} samples loaded successfully")
            
            self.log.emit("Configuring data generators...")
            train_generator, val_generator = create_data_generators(x_train, y_train)
            self.log.emit("Data augmentation initialized")
            
            self.log.emit("Constructing neural network...")
            model = create_model()
            
            train_samples = int(len(x_train) * (1 - VALIDATION_SPLIT))
            val_samples = int(len(x_train) * VALIDATION_SPLIT)
            steps_per_epoch = train_samples // self.config['batch_size']
            validation_steps = val_samples // self.config['batch_size']
            
            history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
            
            self.log.emit("\nTraining protocol initiated\n")
            
            for epoch in range(self.config['epochs']):
                self.log.emit(f"Epoch {epoch + 1}/{self.config['epochs']} ==================")
                
                results = model.fit(
                    train_generator,
                    epochs=1,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=val_generator,
                    validation_steps=validation_steps,
                    verbose=0,
                    callbacks=None
                )
                
                # Update history
                history['loss'].append(results.history['loss'][0])
                history['accuracy'].append(results.history['accuracy'][0])
                history['val_loss'].append(results.history['val_loss'][0])
                history['val_accuracy'].append(results.history['val_accuracy'][0])
                
                self.progress.emit({
                    'epoch': epoch + 1,
                    'history': history
                })
                
                # Log metrics
                self.log.emit(
                    f"Training   - Loss: {results.history['loss'][0]:.4f} | "
                    f"Accuracy: {results.history['accuracy'][0]:.4f}"
                )
                self.log.emit(
                    f"Validation - Loss: {results.history['val_loss'][0]:.4f} | "
                    f"Accuracy: {results.history['val_accuracy'][0]:.4f}"
                )
            
            self.log.emit("\nSaving model state...")
            save_model(model)
            self.log.emit("Training sequence completed successfully.")
            self.finished.emit(history)
            
        except Exception as e:
            self.log.emit(f"Error: Training sequence failed - {str(e)}")

class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.training_thread = None
        self.init_ui()
        
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
        
        config_style = """
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
            QSpinBox, QDoubleSpinBox {
                background-color: #1a1a1a;
                color: #d4d4d4;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 3px;
                font-family: 'Consolas', monospace;
            }
        """
        
        # Left side - Controls and Terminal
        left_side = QWidget()
        left_layout = QVBoxLayout()
        
        # Training Configuration
        config_group = QGroupBox("Model Configuration")
        config_group.setStyleSheet(config_style)
        config_layout = QGridLayout()
        
        config_layout.addWidget(QLabel("Batch Size:"), 0, 0)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(BATCH_SIZE)
        config_layout.addWidget(self.batch_size, 0, 1)
        
        config_layout.addWidget(QLabel("Epochs:"), 1, 0)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(EPOCHS)
        config_layout.addWidget(self.epochs, 1, 1)
        
        config_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.0001, 0.1)
        self.learning_rate.setSingleStep(0.0001)
        self.learning_rate.setValue(LEARNING_RATE)
        config_layout.addWidget(self.learning_rate, 2, 1)
        
        config_group.setLayout(config_layout)
        
        # Training Controls
        controls_layout = QHBoxLayout()
        self.train_button = QPushButton("Initialize Training")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setStyleSheet(button_style)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setStyleSheet(button_style)
        
        controls_layout.addWidget(self.train_button)
        controls_layout.addWidget(self.stop_button)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(progress_style)
        
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
        
        left_layout.addWidget(config_group)
        left_layout.addLayout(controls_layout)
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(QLabel("System Output:"))
        left_layout.addWidget(self.console)
        left_side.setLayout(left_layout)
        
        # Right side - Plots
        right_side = QWidget()
        right_layout = QVBoxLayout()
        
        plt.style.use('dark_background')
        
        # Create accuracy plot
        self.acc_figure, self.acc_ax = plt.subplots(facecolor='#1a1a1a')
        self.acc_canvas = FigureCanvas(self.acc_figure)
        self.acc_ax.set_facecolor('#1a1a1a')
        for spine in self.acc_ax.spines.values():
            spine.set_color('#3d3d3d')
        self.acc_ax.tick_params(colors='#d4d4d4')
        self.acc_ax.set_title('Model Accuracy', color='#d4d4d4', pad=20)
        self.acc_ax.set_xlabel('Epoch', color='#d4d4d4')
        self.acc_ax.set_ylabel('Accuracy', color='#d4d4d4')
        self.acc_ax.grid(True, linestyle='--', alpha=0.3, color='#3d3d3d')
        
        # Create loss plot
        self.loss_figure, self.loss_ax = plt.subplots(facecolor='#1a1a1a')
        self.loss_canvas = FigureCanvas(self.loss_figure)
        self.loss_ax.set_facecolor('#1a1a1a')
        for spine in self.loss_ax.spines.values():
            spine.set_color('#3d3d3d')
        self.loss_ax.tick_params(colors='#d4d4d4')
        self.loss_ax.set_title('Model Loss', color='#d4d4d4', pad=20)
        self.loss_ax.set_xlabel('Epoch', color='#d4d4d4')
        self.loss_ax.set_ylabel('Loss', color='#d4d4d4')
        self.loss_ax.grid(True, linestyle='--', alpha=0.3, color='#3d3d3d')
        
        # Style the plot containers
        plot_container_style = """
            background-color: #1a1a1a;
            border: 1px solid #3d3d3d;
            border-radius: 5px;
            padding: 5px;
        """
        
        for canvas in [self.acc_canvas, self.loss_canvas]:
            canvas_widget = QWidget()
            canvas_layout = QVBoxLayout()
            canvas_layout.addWidget(canvas)
            canvas_widget.setLayout(canvas_layout)
            canvas_widget.setStyleSheet(plot_container_style)
            right_layout.addWidget(canvas_widget)
        
        right_side.setLayout(right_layout)
        
        # Add both sides to main layout
        layout.addWidget(left_side, stretch=60)
        layout.addWidget(right_side, stretch=40)
        
        self.setLayout(layout)
    
    def start_training(self):
        config = {
            'batch_size': self.batch_size.value(),
            'epochs': self.epochs.value(),
            'learning_rate': self.learning_rate.value()
        }
        
        self.progress_bar.setMaximum(config['epochs'])
        self.progress_bar.setValue(0)
        
        self.training_thread = TrainingThread(config)
        self.training_thread.progress.connect(self.update_progress)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.log.connect(lambda msg: self.console.append(msg))
        
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.training_thread.start()
    
    def stop_training(self):
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.training_thread.wait()
            self.console.append("Training sequence terminated by user")
            self.training_finished({})
    
    def update_progress(self, data):
        epoch = data['epoch']
        history = data['history']
        
        self.progress_bar.setValue(epoch)
        
        # Update accuracy plot
        self.acc_ax.clear()
        self.acc_ax.plot(history['accuracy'], label='Training', color='#007acc', linestyle='-')
        self.acc_ax.plot(history['val_accuracy'], label='Validation', color='#60cdff', linestyle='--')
        self.acc_ax.set_title('Model Accuracy', color='#d4d4d4', pad=20)
        self.acc_ax.set_xlabel('Epoch', color='#d4d4d4')
        self.acc_ax.set_ylabel('Accuracy', color='#d4d4d4')
        self.acc_ax.grid(True, linestyle='--', alpha=0.3, color='#3d3d3d')
        self.acc_ax.tick_params(colors='#d4d4d4')
        legend = self.acc_ax.legend(facecolor='#1a1a1a')
        plt.setp(legend.get_texts(), color='#d4d4d4')
        for spine in self.acc_ax.spines.values():
            spine.set_color('#3d3d3d')
        self.acc_canvas.draw()
        
        # Update loss plot
        self.loss_ax.clear()
        self.loss_ax.plot(history['loss'], label='Training', color='#007acc', linestyle='-')
        self.loss_ax.plot(history['val_loss'], label='Validation', color='#60cdff', linestyle='--')
        self.loss_ax.set_title('Model Loss', color='#d4d4d4', pad=20)
        self.loss_ax.set_xlabel('Epoch', color='#d4d4d4')
        self.loss_ax.set_ylabel('Loss', color='#d4d4d4')
        self.loss_ax.grid(True, linestyle='--', alpha=0.3, color='#3d3d3d')
        self.loss_ax.tick_params(colors='#d4d4d4')
        legend = self.loss_ax.legend(facecolor='#1a1a1a')
        plt.setp(legend.get_texts(), color='#d4d4d4')
        for spine in self.loss_ax.spines.values():
            spine.set_color('#3d3d3d')
        self.loss_canvas.draw()
    
    def training_finished(self, history):
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.console.append("\nTraining sequence completed.")