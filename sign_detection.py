import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import Counter
import os
import time
import string
import requests
import pyttsx3
import speech_recognition as sr
import tkinter as tk
from PIL import Image, ImageTk
from itertools import count
from tkinter import messagebox
import Levenshtein

# Configuration for sign detection
WEIGHT_HAND = 1.0
WEIGHT_FACE = 0.1
WEIGHT_POSE = 0.3
FACE_LANDMARKS_TO_USE = [1, 4, 10]
BUFFER_LENGTH = 15
CONFIDENCE_THRESHOLD = 0.6
VOTE_THRESHOLD = 0.6

# Load model and scaler for sign detection
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']
scaler = model_dict['scaler']

# Label mapping for sign detection
labels_dict = {
    0: 'ok/correct', 1: 'good', 2: 'two', 3: 'engineer',
    4: 'Nice to meet you', 5: 'alright', 6: 'God', 7: 'Walk',
    8: 'sorry', 9: 'call', 10: 'here', 11: 'light', 12: 'namaste/pray', 
    13: 'read', 14: 'loss/headache', 15: 'bond/love', 16: 'strength/powerful',
    17: 'may i go to washroom', 18: 'flower', 19: 'waiting for your action',
    20: 'drink/water', 21: 'food', 22: 'computer science',
    23: 'engineer', 24: 'how are you?', 25: 'i am fine',
    26: 'promise', 27: 'yesterday', 28: 'today',
    29: 'tomorrow' , 30:'project' , 31 : 'professional' ,32:'lab'  , 33: 'you'
}

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ----------------------- GUI Functions ----------------------- #

def center_window(root, width, height):
    """Center the tkinter window on the screen."""
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_coordinate = (screen_width - width) // 2
    y_coordinate = (screen_height - height) // 2
    root.geometry(f"{width}x{height}+{x_coordinate}+{y_coordinate}")

def custom_buttonbox(msg, image, choices):
    """Create a custom button box with image and message."""
    root = tk.Tk()
    root.title("Sign Language Assistant")
    
    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)
    
    try:
        img = tk.PhotoImage(file=image)
        img_label = tk.Label(frame, image=img)
        img_label.image = img
        img_label.pack(pady=10)
    except tk.TclError as e:
        messagebox.showerror("Error", f"Failed to load image: {e}")
        root.destroy()
        return
    
    msg_label = tk.Label(frame, text=msg, font=("Arial", 16, "bold"))
    msg_label.pack(pady=10)
    
    buttonbox = tk.Frame(frame)
    buttonbox.pack(pady=10)
    
    for choice in choices:
        button = tk.Button(buttonbox, text=choice, 
                           command=lambda c=choice: on_button_click(root, c),
                           width=15, height=2, font=("Arial", 12))
        button.pack(side=tk.LEFT, padx=5)
    
    center_window(root, 800, 550)
    root.mainloop()

def on_button_click(root, choice):
    """Handle button clicks in the main menu."""
    root.destroy()
    if choice == "Voice To Sign":
        speech_to_sign()
    elif choice == "Sign Detection":
        signDetection()
    elif choice == "Exit":
        quit()

# ----------------------- Speech to Sign Functions ----------------------- #

def find_closest_match(input_text, gesture_list):
    """Find the closest match between input text and available gestures."""
    min_distance = float('inf')
    closest_match = None
    
    for gesture in gesture_list:
        distance = Levenshtein.distance(input_text, gesture)
        if distance < min_distance:
            min_distance = distance
            closest_match = gesture
            
    return closest_match if min_distance / max(len(input_text), len(closest_match)) < 0.4 else None

def display_alphabets(text, alphabet_list):
    """Display individual alphabets of the unrecognized string sequentially in a thumbnail gallery."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Gallery</title>
        <style>
            .gallery {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 10px;
            }
            .thumbnail {
                width: 200px;
                height: 200px;
            }
            .new-line {
                flex-basis: 100%;
                height: 0;
                margin: 0;
                padding: 0;
            }
        </style>
    </head>
    <body>
        <div class="gallery">
    """
    
    for char in text:
        if char != ' ':
            if char in alphabet_list:
                image_path = f"letters/{char}.jpg"
            else:
                image_path = "letters/empty.jpg"
            html_content += f'<img class="thumbnail" src="{image_path}" alt="{char}">'
        else:
            html_content += '<div class="new-line"></div>'
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open("image_gallery.html", "w") as html_file:
        html_file.write(html_content)
    
    os.system("start image_gallery.html")

class ImageLabel(tk.Label):
    """A label that displays images, and plays them if they are gifs."""
    
    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        self.loc = 0
        self.frames = []
        
        try:
            for i in count(1):
                self.frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass
        
        try:
            self.delay = im.info['duration']
        except:
            self.delay = 100
        
        if len(self.frames) == 1:
            self.config(image=self.frames[0])
        else:
            self.next_frame()
    
    def unload(self):
        self.config(image=None)
        self.frames = None
    
    def next_frame(self):
        if self.frames:
            self.loc += 1
            self.loc %= len(self.frames)
            self.config(image=self.frames[self.loc])
            self.after(self.delay, self.next_frame)

def speech_to_sign():
    """Convert speech to sign language."""
    r = sr.Recognizer()
    isl_gif = ['any questions', 'are you angry', 'are you busy', 'are you hungry', 'be careful',
                'did you book tickets', 'did you finish homework', 'do you have money', 
                'do you want something to drink', 'do you want tea or coffee', 'do you watch TV',
                'dont worry', 'flower is beautiful', 'good afternoon', 'good evening', 'good morning',
                'good question', 'happy journey', 'hello what is your name',
                'how many people are there in your family', 'i am a clerk', 'i am bore doing nothing',
                'i am fine', 'i am sorry', 'i am thinking', 'i am tired', 'i dont understand anything', 
                'i go to a theatre', 'i love to shop', 'i had to say something but I forgot', 
                'i like pink colour', 'i live in nagpur', 'lets go for lunch', 'nice to meet you',
                'open the door', 'please call me later', 'please use dustbin dont throw garbage', 
                'please wait for sometime', 'shall I help you', 'shall we go together tommorow', 
                'sign language interpreter', 'sit down', 'stand up', 'take care', 'there was traffic jam', 
                'wait I am thinking', 'what are you doing', 'what is the problem',
                'what is todays date', 'what is your father do', 'what is your job', 
                'what is your mobile number', 'what is your name', 'whats up', 'when is your interview', 
                'when we will go', 'where do you stay', 'where is the bathroom', 
                'where is the police station', 'you are wrong', 'address', 'agra', 'ahmedabad',
                'all', 'april', 'assam', 'august', 'australia', 'badoda', 'banana', 'banaras', 
                'bangalore', 'bihar', 'bridge', 'cat', 'chandigarh', 'chennai', 'christmas', 
                'church', 'clinic', 'coconut', 'crocodile', 'dasara', 'deaf', 'december', 'deer', 
                'delhi', 'dollar', 'duck', 'february', 'friday', 'fruits', 'glass',
                'grapes', 'hello', 'hindu', 'hyderabad', 'india', 'january', 'jesus', 'job', 'july',
                'karnataka', 'kerala', 'krishna', 'litre', 'mango', 'may', 'mile', 'monday', 
                'mumbai', 'museum', 'muslim', 'nagpur', 'october', 'orange', 'pakistan', 'pass', 
                'police station', 'post office', 'pune', 'punjab', 'rajasthan', 'ram', 'restaurant', 
                'saturday', 'september', 'shop', 'sleep', 'south africa', 'story',
                'sunday', 'tamil nadu', 'temperature', 'temple', 'thursday', 'toilet', 'tomato', 
                'town', 'tuesday', 'usa', 'village', 'voice', 'wednesday', 'weight', 
                'please wait for sometime', 'what is your mobile number', 'what are you doing', 
                'are you busy']
    
    arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 
           's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        r.adjust_for_ambient_noise(source)
        print("Listening... Speak now!")
        
        try:
            audio = r.listen(source, timeout=5)
            print("Processing speech...")
            
            a = r.recognize_google(audio)
            a = a.lower()
            print('You Said: ' + a.lower())
            
            closest_match = find_closest_match(a, isl_gif)
            
            if closest_match:
                print(f"Closest match found: {closest_match}")
                a = closest_match.lower()
            
            for c in string.punctuation:
                a = a.replace(c, "")
            
            if a.lower() == 'goodbye' or a.lower() == 'good bye' or a.lower() == 'bye':
                print("Oops! Time to say goodbye")
                messagebox.showinfo("Goodbye", "Returning to main menu...")
                time.sleep(1)
                main()
                return
            
            elif a.lower() in isl_gif:
                root = tk.Tk()
                root.title("Sign Language Visualization")
                center_window(root, 500, 500)
                
                lbl = ImageLabel(root)
                lbl.pack()
                
                try:
                    lbl.load(f'ISL_Gifs/{a.lower()}.gif')
                    
                    back_button = tk.Button(root, text="Back to Main Menu", 
                                           command=lambda: [root.destroy(), main()],
                                           width=15, height=1, font=("Arial", 10))
                    back_button.pack(pady=10)
                    
                    root.mainloop()
                except Exception as e:
                    messagebox.showerror("Error", f"Could not load GIF: {e}")
                    root.destroy()
                    main()
            
            else:
                messagebox.showinfo("Information", 
                                   "Showing individual letter signs for: " + a)
                display_alphabets(a, arr)
                time.sleep(5)
                main()
            
        except sr.UnknownValueError:
            messagebox.showerror("Error", "Speech Recognition could not understand audio")
            main()
        except sr.RequestError as e:
            messagebox.showerror("Error", 
                               f"Could not request results from Google Speech Recognition service; {e}")
            main()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            main()

# ----------------------- Sign Detection Functions ----------------------- #

# Draw colored finger connections
def draw_fingers(hand_landmarks, frame, handedness='Left'):
    """Draw colored finger connections on the frame."""
    FINGER_COLORS = [(255, 0, 0), (0, 255, 0), (255, 255, 0),
                    (255, 0, 255), (0, 255, 255)]
    
    fingers = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20]
    ]
    
    h, w, _ = frame.shape
    for i, finger in enumerate(fingers):
        color = FINGER_COLORS[i]
        for j in range(len(finger) - 1):
            pt1 = hand_landmarks.landmark[finger[j]]
            pt2 = hand_landmarks.landmark[finger[j+1]]
            x1, y1 = int(pt1.x * w), int(pt1.y * h)
            x2, y2 = int(pt2.x * w), int(pt2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (x1, y1), 3, color, -1)

# Extract weighted features from landmarks
def get_weighted_features(results):
    """Extract weighted features from landmarks for model prediction."""
    features = []
    # Left hand
    if results.left_hand_landmarks:
        xs = [lm.x for lm in results.left_hand_landmarks.landmark]
        ys = [lm.y for lm in results.left_hand_landmarks.landmark]
        minx, miny = min(xs), min(ys)
        for lm in results.left_hand_landmarks.landmark:
            features += [(lm.x - minx) * WEIGHT_HAND, (lm.y - miny) * WEIGHT_HAND]
    else:
        features += [0.0] * 42
    # Right hand
    if results.right_hand_landmarks:
        xs = [lm.x for lm in results.right_hand_landmarks.landmark]
        ys = [lm.y for lm in results.right_hand_landmarks.landmark]
        minx, miny = min(xs), min(ys)
        for lm in results.right_hand_landmarks.landmark:
            features += [(lm.x - minx) * WEIGHT_HAND, (lm.y - miny) * WEIGHT_HAND]
    else:
        features += [0.0] * 42
    # Face landmarks
    if results.face_landmarks:
        sel_x = [results.face_landmarks.landmark[i].x for i in FACE_LANDMARKS_TO_USE]
        sel_y = [results.face_landmarks.landmark[i].y for i in FACE_LANDMARKS_TO_USE]
        minx, miny = min(sel_x), min(sel_y)
        for i in FACE_LANDMARKS_TO_USE:
            lm = results.face_landmarks.landmark[i]
            features += [(lm.x - minx) * WEIGHT_FACE, (lm.y - miny) * WEIGHT_FACE]
    else:
        features += [0.0] * (len(FACE_LANDMARKS_TO_USE) * 2)
    # Pose landmarks
    if results.pose_landmarks:
        xs = [lm.x for lm in results.pose_landmarks.landmark]
        ys = [lm.y for lm in results.pose_landmarks.landmark]
        minx, miny = min(xs), min(ys)
        for lm in results.pose_landmarks.landmark:
            features += [(lm.x - minx) * WEIGHT_POSE, (lm.y - miny) * WEIGHT_POSE]
    else:
        features += [0.0] * 66
    return features

# Draw a semi-transparent overlay and text
def draw_overlay(frame, text, ypos=80):
    """Draw a semi-transparent overlay with wrapped text on the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    max_width = frame.shape[1] - 40  # 20px margin on each side
    
    # Split text into words
    words = text.split(' ')
    lines = []
    current_line = words[0]
    
    # Create lines based on fitting text within max_width
    for word in words[1:]:
        test_line = current_line + ' ' + word
        (w, h), _ = cv2.getTextSize(test_line, font, scale, thickness)
        if w <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)  # Add the last line
    
    # Calculate total overlay height
    line_height = h + 10
    total_height = len(lines) * line_height
    
    # Draw background overlay
    x, y = 20, ypos
    overlay = frame.copy()
    cv2.rectangle(overlay, (x-10, y-line_height), (x+max_width, y+total_height-line_height), 
                 (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw each line of text
    for i, line in enumerate(lines):
        line_y = y + i * line_height
        cv2.putText(frame, line, (x, line_y), font, scale, (255, 255, 255), 
                   thickness, cv2.LINE_AA)

# Main detection function
def signDetection():
    """Main function for sign language detection using MediaPipe Holistic approach."""
    cap = cv2.VideoCapture(0)
    buffer = []
    recent_predictions = []
    final_predictions = []
    tts = pyttsx3.init()
    
    # Create a resizable window for better display
    cv2.namedWindow("Sign Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sign Detection", 1024, 768)  # Larger display size
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as holistic:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            
            # Draw landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.face_landmarks:
                h, w, _ = frame.shape
                for i in FACE_LANDMARKS_TO_USE:
                    pt = results.face_landmarks.landmark[i]
                    cv2.circle(frame, (int(pt.x*w), int(pt.y*h)), 3, (0,255,255), -1)
            if results.left_hand_landmarks:
                draw_fingers(results.left_hand_landmarks, frame)
            if results.right_hand_landmarks:
                draw_fingers(results.right_hand_landmarks, frame)
            
            # Feature buffering
            features = get_weighted_features(results)
            buffer.append(features)
            if len(buffer) > BUFFER_LENGTH:
                buffer = buffer[-BUFFER_LENGTH:]
            
            # Prediction and voting
            if len(buffer) >= BUFFER_LENGTH:
                mean_feats = np.mean(buffer, axis=0).reshape(1, -1)
                try:
                    scaled = scaler.transform(mean_feats)
                    prediction = model.predict(scaled)
                    probabilities = model.predict_proba(scaled)[0]
                    pred_idx = np.argmax(probabilities)
                    confidence = probabilities[pred_idx]
                    
                    if confidence >= CONFIDENCE_THRESHOLD:
                        recent_predictions.append(int(prediction[0]))
                        if len(recent_predictions) > 5:
                            recent_predictions.pop(0)
                        if recent_predictions:
                            top_class, vote_count = Counter(recent_predictions).most_common(1)[0]
                            if vote_count / len(recent_predictions) >= VOTE_THRESHOLD:
                                label = labels_dict.get(top_class, "Unknown")
                                final_predictions.append(label)
                                cv2.putText(frame, f"{label} ({confidence:.2f})", 
                                          (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                          1.2, (0, 0, 255), 3, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "Waiting for clear gesture...", 
                                  (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                  1.0, (0, 255, 255), 2, cv2.LINE_AA)
                except Exception as e:
                    print("Prediction error:", e)
                    cv2.putText(frame, f"Error: {str(e)}", 
                              (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Instruction overlay
            cv2.putText(frame, "Press G to interpret sequence | B to go back", 
                      (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Display signs detected so far
            if final_predictions:
                signs_text = f"Signs detected: {', '.join(final_predictions[-5:])}"
                cv2.putText(frame, signs_text, 
                          (20, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
                          0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow("Sign Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            
            # On 'g', call Gemini
            if key == ord('g') and final_predictions:
                prompt = (
                    "You are an expert at converting sequences of raw sign-language labels into fluent, human-readable English. "
                    "You are helping a poor person who cant speak or hear communicate with normal people. "
                    f"Given this list of words: {final_predictions}, return one single paragraph adding all the necessary context "
                    "relatable to the words capturing their meaning. You are supposed to put emotions into the words, "
                    "since they cant be captured by the words."
                )
                try:
                    # Show processing message
                    processing_frame = frame.copy()
                    cv2.putText(processing_frame, "Processing... Please wait", 
                              (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.8, (0, 165, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Sign Detection", processing_frame)
                    cv2.waitKey(1)  # Update display
                    
                    resp = requests.post(
                        # "http://localhost:8000/gemini",
                        "https://sign-language-3-5vax.onrender.com/gemini",
                        json={"message": prompt}
                    )
                    resp.raise_for_status()
                    ai_text = resp.json().get("response", "")
                    
                    # Create a new window for the interpretation
                    result_frame = frame.copy()
                    cv2.putText(result_frame, "INTERPRETATION (Press any key to continue)", 
                               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
                    draw_overlay(result_frame, ai_text, ypos=80)
                    
                    # Speak the text in background
                    tts.say(ai_text)
                    tts.runAndWait()
                    
                    # Show result in a dedicated window
                    cv2.namedWindow("Sign Interpretation", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Sign Interpretation", 1024, 768)
                    cv2.imshow("Sign Interpretation", result_frame)
                    cv2.waitKey(0)  # Wait for key press
                    cv2.destroyWindow("Sign Interpretation")
                    
                except Exception as e:
                    print("Error calling Gemini API:", e)
                    error_frame = frame.copy()
                    cv2.putText(error_frame, f"API Error: {str(e)}", 
                              (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Sign Detection", error_frame)
                    cv2.waitKey(2000)  # Show error for 2 seconds
                finally:
                    final_predictions.clear()
            
            # Back to main menu
            if key == ord('b'):
                break
                
            # Quit
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                quit()
    
    cap.release()
    cv2.destroyAllWindows()
    main()

def main():
    """Main function to run the application."""
    image_path = "logo.png"
    message = "SIGN LANGUAGE ASSISTANT FOR HEARING IMPAIRMENTS"
    choices = ["Voice To Sign", "Sign Detection", "Exit"]
    
    custom_buttonbox(message, image_path, choices)

if __name__ == '__main__':
    main()