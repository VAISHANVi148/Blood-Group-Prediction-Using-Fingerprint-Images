import tkinter as tk
from tkinter import ttk
from tkvideo import tkvideo
import subprocess  # Import subprocess module

# Function to switch to the main application page
def show_main_page():
    # Run the GUI_main.py file
    subprocess.Popen(['python', 'GUI_Main.py'])  # Adjust the command as needed for your environment

# Initialize the main window
root = tk.Tk()
root.title("Detection of Blood Groups Using Fingerprint Images")

# Full-screen configuration
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.configure(background="#EAEAEA")  # Light gray background for a modern look

# Header Section
header_frame = tk.Frame(root, bg="#0056A0", height=100)
header_frame.pack(fill="x")

header_label = tk.Label(header_frame, text="Detection of Blood Groups Using Fingerprint Images",
                        fg="white", bg="#0056A0", font=("Arial", 26, "bold"))
header_label.pack(pady=20)

# Body Section
body_frame = tk.Frame(root, bg="#EAEAEA", height=h - 200)
body_frame.pack(fill="both", expand=True, pady=20)

# Left and Right Padding
body_frame.columnconfigure(0, weight=1)
body_frame.columnconfigure(3, weight=1)

# Video Display
video_label = tk.Label(body_frame, bg="white")
video_label.grid(row=0, column=1, columnspan=2, padx=20, pady=20)

# Use tkvideo to play video with an adjusted size
player = tkvideo("1.mp4", video_label, loop=1, size=(int(w * 0.6), int(h * 0.4)))
player.play()

# Welcome Label
welcome_label = tk.Label(body_frame, text="Welcome to Blood Group Detection System",
                         font=("Arial", 22, "bold"), bg="#EAEAEA", fg="#333333", pady=20)
welcome_label.grid(row=1, column=1, columnspan=2, sticky="n")

# Cards Section
card_frame = tk.Frame(body_frame, bg="#EAEAEA")
card_frame.grid(row=2, column=1, columnspan=2, padx=20, pady=20, sticky="n")

# Card Style
card_style = {
    "bg": "#FFFFFF",
    "bd": 0,
    "relief": "flat",
    "width": 200,
    "height": 200
}

# First Card
card1 = tk.Frame(card_frame, **card_style)
card1.grid(row=0, column=0, padx=20, pady=20)
card1_label = tk.Label(card1, text="Advanced Detection", bg="#FFFFFF",
                       font=("Arial", 16, "bold"), fg="#0056A0")
card1_label.pack(pady=10)
card1_desc = tk.Label(card1, text="Using fingerprint images to detect blood groups.", bg="#FFFFFF",
                      font=("Arial", 12), fg="#666666")
card1_desc.pack(pady=5)

# Second Card
card2 = tk.Frame(card_frame, **card_style)
card2.grid(row=0, column=1, padx=20, pady=20)
card2_label = tk.Label(card2, text="Accurate Results", bg="#FFFFFF",
                       font=("Arial", 16, "bold"), fg="#0056A0")
card2_label.pack(pady=10)
card2_desc = tk.Label(card2, text="Reliable and accurate blood group detection.", bg="#FFFFFF",
                      font=("Arial", 12), fg="#666666")
card2_desc.pack(pady=5)

# Third Card
card3 = tk.Frame(card_frame, **card_style)
card3.grid(row=0, column=2, padx=20, pady=20)
card3_label = tk.Label(card3, text="Easy to Use", bg="#FFFFFF",
                       font=("Arial", 16, "bold"), fg="#0056A0")
card3_label.pack(pady=10)
card3_desc = tk.Label(card3, text="User-friendly system for quick detection.", bg="#FFFFFF",
                      font=("Arial", 12), fg="#666666")
card3_desc.pack(pady=5)

# Start Button
def on_enter(e):
    start_button['background'] = '#004080'  # Darker on hover

def on_leave(e):
    start_button['background'] = '#0056A0'  # Lighter when not hovered

start_button = tk.Button(body_frame, text="Start", command=show_main_page,
                         width=16, height=1, bg="#0056A0", fg="white", font=("Arial", 16, "bold"),
                         relief="raised", bd=0, cursor="hand2")
start_button.grid(row=3, column=1, columnspan=2, pady=18)
start_button.bind("<Enter>", on_enter)
start_button.bind("<Leave>", on_leave)

# Footer Section
footer_frame = tk.Frame(root, bg="#0056A0", height=100)
footer_frame.pack(fill="x", side="bottom")

footer_label = tk.Label(footer_frame, text="Developed by Team XYZ | 2024", bg="#0056A0", fg="white",
                        font=("Arial", 12))
footer_label.pack(pady=20)

# Start the main loop
root.mainloop()
