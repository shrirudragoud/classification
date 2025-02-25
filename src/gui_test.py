import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from model import load_model_from_checkpoint
from config import *

class CatDogClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat vs Dog Classifier")
        self.root.geometry("800x600")
        
        # Load model once at startup
        print("Loading model...")
        self.model = load_model_from_checkpoint()
        if self.model is None:
            raise Exception("Could not load model")
        print("Model loaded successfully!")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Select image button
        self.select_btn = ttk.Button(
            main_frame, 
            text="Select Image", 
            command=self.select_image
        )
        self.select_btn.grid(row=0, column=0, pady=10)
        
        # Image display area
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=1, column=0, pady=10)
        
        # Prediction result
        self.result_var = tk.StringVar(value="Select an image to see prediction")
        self.result_label = ttk.Label(
            main_frame,
            textvariable=self.result_var,
            font=("Arial", 14)
        )
        self.result_label.grid(row=2, column=0, pady=10)
        
        # Confidence bar
        self.confidence_frame = ttk.Frame(main_frame)
        self.confidence_frame.grid(row=3, column=0, pady=10)
        
        self.confidence_bar = ttk.Progressbar(
            self.confidence_frame,
            length=300,
            mode='determinate'
        )
        self.confidence_bar.grid(row=0, column=0, padx=5)
        
        self.confidence_label = ttk.Label(
            self.confidence_frame,
            text="0%"
        )
        self.confidence_label.grid(row=0, column=1, padx=5)
        
    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.process_image(file_path)
    
    def process_image(self, image_path):
        try:
            # Load and display image
            display_image = Image.open(image_path)
            # Resize for display while maintaining aspect ratio
            display_image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(display_image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep a reference
            
            # Preprocess image for prediction
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Failed to load image")
                
            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize to match model's expected input size
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Add channel dimension and batch dimension
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            img = np.expand_dims(img, axis=0)   # Add batch dimension
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            # Make prediction
            prediction = self.model.predict(img, verbose=0)[0][0]
            
            # Use a lower threshold for dog detection (0.4 instead of 0.5)
            DOG_THRESHOLD = 0.4
            
            # Calculate confidence based on threshold
            confidence = prediction if prediction >= DOG_THRESHOLD else 1 - prediction
            confidence_pct = confidence * 100
            
            # Update result with adjusted threshold
            result = "Dog" if prediction >= DOG_THRESHOLD else "Cat"
            self.result_var.set(f"Prediction: {result}")
            
            # Update confidence bar
            self.confidence_bar['value'] = confidence_pct
            self.confidence_label['text'] = f"{confidence_pct:.1f}%"
            
        except Exception as e:
            self.result_var.set(f"Error: {str(e)}")
            self.confidence_bar['value'] = 0
            self.confidence_label['text'] = "0%"
        
        # Make prediction
        prediction = self.model.predict(img, verbose=0)[0][0]
        
        # Calculate confidence
        confidence = prediction if prediction >= 0.5 else 1 - prediction
        confidence_pct = confidence * 100
        
        # Update result
        result = "Dog" if prediction >= 0.5 else "Cat"
        self.result_var.set(f"Prediction: {result}")
        
        # Update confidence bar
        self.confidence_bar['value'] = confidence_pct
        self.confidence_label['text'] = f"{confidence_pct:.1f}%"

def main():
    root = tk.Tk()
    app = CatDogClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()