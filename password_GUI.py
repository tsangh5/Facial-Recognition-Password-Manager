import tkinter as tk
from tkinter import font as tkFont
from tkinter import messagebox
import cv2
import numpy as np
from time import sleep
from PIL import Image
import uuid
import sys, os, json

TRAINING_FILE = "training_file.yml"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MIN_DETECT_WIDTH = 40
MIN_DETECT_HEIGHT = 40
IMAGES_FOLDER = 'Images'
CASCADE_FILE = 'haarcascade_frontalface_default.xml'
recognise = True
not_recognised_count = 0
face_detected = True
if not os.path.exists("names"):
    open('names', 'w+')
if not os.path.exists("passwords"):
    open('passwords', 'w+')

class PasswordApp:
    def __init__(self, app):
        self.app = app
        self.window = tk.Toplevel()
        self.window.geometry("700x1000")  # sets the tkinter window to 700 pixels wide and 1000 pixels in height
        self.window.title('Password Storage')  # gives the tkinter GUI the names 'Password Storage'
        self.window.protocol("WM_DELETE_WINDOW", self.app.quit)  # Makes the close button end the program
        helv36 = tkFont.Font(family='Helvetica', size=36, weight=tkFont.BOLD)  # gives a font 'helv36' those parameters
        self.site_entry_instructions = tk.Label(self.window, text='WEB LINK/NAME')  # creates a label with that text
        self.site_entry_instructions.place(x=30, y=20)  # places the label at  coordinates (30, 205)
        self.pw_entry_instructions = tk.Label(self.window, text='PASSWORD')  # creates label with that text
        self.pw_entry_instructions.place(x=260, y=20)  # places the label at coordinates (260, 20)
        self.site_entry = tk.Entry(self.window)  # creates an entry box
        self.site_entry.place(x=30, y=40)  # places the entry box at coordinates (30, 40)
        self.site_entry['width'] = 30  # gives the entry box a width of 30
        self.pw_entry = tk.Entry(self.window)  # creates an entry box
        self.pw_entry.place(x=260, y=40)  # places the entry box at coordinates (30, 40)
        self.pw_entry['width'] = 30  # gives the entry box a width of 30
        self.add_button = tk.Button(self.window, text='+', font=helv36, command=self.add, bg='#A9A9A9')  # creates a
        # button with those parameters
        self.add_button.place(x=500, y=20)  # places the button at coordinates (500, 20)
        self.add_button['height'] = 1  # gives the button height of 1
        self.add_button['width'] = 5  # gives the button width of 5
        self.site_list = tk.Listbox(self.window, width=30, height=30)  # creates a listbox
        self.site_list.place(x=30, y=80)  # places the listbox at coordinates (30, 80)
        self.pw_list = tk.Listbox(self.window, width=30, height=30)  # creates a listbox
        self.pw_list.place(x=260, y=80)  # places the listbox at coordinates (260,80)
        with open("names", "r") as f:  # opens file 'names' for reading
            content = f.read()
            lines = content.splitlines()  # splits each line from the file
            for item in lines:
                self.site_list.insert(tk.END, item)  # inserts each line into the list box
        with open('passwords', "r") as f:  # opens file 'passwords' for reading
            content = f.read()
            lines = content.splitlines()  # splits each line from the file
            for item in lines:
                self.pw_list.insert(tk.END, item)  # inserts each line into the list box

    def add(self):
        site = self.site_entry.get()
        pw = self.pw_entry.get()
        self.site_list.insert(tk.END, site)
        self.pw_list.insert(tk.END, pw)
        self.site_entry.delete(0, tk.END)
        self.pw_entry.delete(0, tk.END)
        with open("names", "a") as f:
            f.write(site)
        with open('passwords', 'a') as f:
            f.write(pw)


def convert_cv2_to_pil(cv2_image):
    return Image.formarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY))


def convert_pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


class Camera():
    def __init__(self, camera_device_id=0, width=1280, height=720):
        self.camera_device_id = camera_device_id
        self.camera_width = 1280
        self.camera_height = 720
        self.cap = cv2.VideoCapture(self.camera_device_id)
        self.cap.set(3, self.camera_width)
        self.cap.set(4, self.camera_height)

    def record_video(self, length=0.0, filename="", per_frame_callback=None, preview=False):
        pass

    def record_video_stop(self):
        pass

    def take_photo(self, preview=False):
        # Read image from the camera
        ret, img = self.cap.read()
        if preview:
            cv2.imshow(img)
            # Convert from CV2 image to PIL image
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


def get_faces(img, cascade_file):
    if not os.path.exists(cascade_file):
        raise Exception("[get_faces] Cascade file does not exist")
    if not isinstance(img, Image.Image):
        raise Exception("[get_faces] Not a PIL.Image.Image object")
    # Convert from PIL image to CV2 image
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # Returns the colour of face, grayscale of face, and full image containing face if there is a face in the photo
    cascade = cv2.CascadeClassifier(cascade_file)
    # Convert image to grey scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect any faces in the image? Put in an array
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(MIN_DETECT_WIDTH, MIN_DETECT_HEIGHT)
    )
    # If there is a face
    return faces


### Demonstration of functionality

def record_faces():
    # Check the folders exist
    if not os.path.exists("images"):
        os.mkdir("images")
    if not os.path.exists(f"images/my_images"):
        os.mkdir(f"images/my_images")
    # Open the camera
    number_of_images_in_folder = len([name for name in os.listdir('.')])
    camera = Camera(0)
    number_photos_needed = 50 - number_of_images_in_folder
    i = 0
    while i < number_photos_needed + 1:
        # Take a photo
        photo = camera.take_photo()
        # Any faces in the photo?
        faces = get_faces(photo, CASCADE_FILE)
        if faces is not None and len(faces) > 0:
            for (x, y, w, h) in faces:
                # Extract the face portion of the image
                face_image = photo.crop((x, y, x + w, y + h))
                # Save the face image
                filename = f"images/my_images/{i}.jpg"
                face_image.save(filename, "jpeg")
                print(i)
                i += 1


def train_from_faces(training_file):
    """
    Will analyse all the faces in the object's images folder.
    Depending on the number of images this could take some time (allow 10 seconds per 100 images).
    Updates the object's training_file with the resulting calculations for use `recognise_face` function.
    """
    # Path for face image database
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # Find all people to train on
    folders = os.listdir("images")
    # Delete any non-folders
    folders = [f for f in folders if os.path.isdir("Images/" + f)]
    # Create our data storage lists
    face_images = []
    face_numbers = []
    face_number = 0
    for folder in folders:
        print(f"Processing {folder}...")
        for i in range(1, 42):
            # Load the image file, convert it to grayscale
            pimage = Image.open(f"images/{folder}/{i}.jpg").convert('L')
            # Convert to numpy array
            nimage = np.array(pimage, 'uint8')
            face_images.append(nimage)
            face_numbers.append(face_number)  # The folder should be named for the person
        face_number += 1
    # Train with those faces
    recognizer.train(face_images, np.array(face_numbers))
    # Save the model into trainer yml data file
    recognizer.write(TRAINING_FILE)  # recognizer.save() worked on Mac, but not on Pi
    # Return the number of faces trained
    return len(np.unique(face_numbers))


def recognise_faces(training_file):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(training_file)
    # Find all folders we trained on
    folders = os.listdir("images")
    # Delete any non-folders
    folders = [f for f in folders if os.path.isdir("images/" + f)]
    # Take a photo
    camera = Camera(0)
    photo = camera.take_photo()
    # Any faces in the photo?
    faces = get_faces(photo, CASCADE_FILE)
    if faces is not None and len(faces) > 0:
        for (x, y, w, h) in faces:
            # Extract the face portion of the image
            face_image = photo.crop((x, y, x + w, y + h))
            gray = cv2.cvtColor(convert_pil_to_cv2(face_image), cv2.COLOR_BGR2GRAY)
            id, confidence = recognizer.predict(gray)
            # If confidence is less then 100, deem the person recognised (0 == perfect match)
            if confidence < 100:
                person = folders[id]
                recognise = True
            else:
                person = "unknown"
                recognise = False
    else:
        print(f"No faces detected\n")
        face_detected = False
        recognise = False
    return recognise


if not os.path.exists(TRAINING_FILE):
    record_faces()
    train_from_faces(TRAINING_FILE)
else:
    recognise = recognise_faces(TRAINING_FILE)
if recognise:
    if __name__ == "__main__":
        root = tk.Tk()  # Initialise the tk system into an object called `root`
        root.withdraw()  # Hide the default window
        app = PasswordApp(root)  # Run our window, called AppWindow
        root.mainloop()
elif face_detected:
    messagebox.showwarning('Problem', 'Face not recognised!')
    not_recognised_count += 1
    if not_recognised_count > 5:
        messagebox.showwarning('Problem' 'Failed to enter more than 5 times. Program self-destructing......')
        sys.exit()
else:
    sleep(3)
    messagebox.showwarning('Problem', 'No face detected')
    recognise_faces(TRAINING_FILE)

