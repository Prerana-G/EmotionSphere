import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk


emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def preprocess_image(image_path):
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    
    
    image = cv2.resize(image, (48, 48))
    
    
    image = image / 255.0
    
    
    image = np.expand_dims(image, axis=0)  
    image = np.expand_dims(image, axis=-1)  
    return image

def predict_emotion(image_path, model_path='fer_cnn_final_model.h5'):
    """
    Predict the emotion from a given image using the trained model.
    """
    # Load the trained model
    model = load_model(model_path)

    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Predict the emotion
    predictions = model.predict(processed_image)
    
    # Get the predicted class with the highest confidence
    predicted_class = np.argmax(predictions[0])  # Get the index of the highest score

    # Map the predicted class to the emotion label
    predicted_emotion = emotion_labels[predicted_class]
    return predicted_emotion

def open_file_dialog():
    """
    Open a file dialog for the user to select an image file.
    """
    
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp")])
    
    
    if file_path:
        try:
            predicted_emotion = predict_emotion(file_path)
            update_emotion_label(predicted_emotion)  
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def update_emotion_label(predicted_emotion):
    """
    Updates the emotion label in the main window to show the predicted emotion.
    """
    emotion_label_in_card.config(text=f"Predicted Emotion: {predicted_emotion.capitalize()} 🌟") 


root = tk.Tk()
root.title("EmotionSphere - Mood Analyzer 🌈")
root.geometry("600x500")  
root.configure(bg="#f0f8ff")  

label = tk.Label(
    root,
    text="Welcome to EmotionSphere! 😊",
    font=("Helvetica", 24, "bold"),
    bg="#f0f8ff",
    fg="#00008b"
)
label.pack(pady=40)

sub_label = tk.Label(
    root,
    text="See your feelings in pixels, understand your moods with AI!",
    font=("Helvetica", 14, "italic"),
    bg="#f0f8ff",
    fg="#333333"
)
sub_label.pack(pady=10)

sub_label = tk.Label(
    root,
    text="Analyze your mood now! 🧠💖",
    font=("Helvetica", 16),
    bg="#f0f8ff",
    fg="#333333"
)
sub_label.pack(pady=20)

style = ttk.Style()
style.configure("TButton", font=("Helvetica", 14), padding=10)

button = ttk.Button(
    root,
    text="Browse Image 📸",
    command=open_file_dialog
)
button.pack(pady=20)

card_frame = tk.Frame(
    root,
    bg="white",  
    relief="solid",  
    bd=0,  
    padx=20,  
    pady=20   
)
card_frame.pack(pady=40, padx=20, fill="none", expand=False)  


emotion_label_in_card = tk.Label(
    card_frame,
    text="Predicted Emotion: None 😶",  
    font=("Helvetica", 18, "bold"),
    bg="white",
    fg="#00008b"
)
emotion_label_in_card.pack(pady=10)  


footer = tk.Label(
    root,
    text="Developed with ❤️by Prerana",
    font=("Helvetica", 10),
    bg="#f0f8ff",
    fg="#555555"
)
footer.pack(side="bottom", pady=10)


root.mainloop()
