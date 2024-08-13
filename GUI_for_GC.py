import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img

# Load the model
model = load_model('D:/Project/garbage_classifier.h5', compile=False)

# Define the class labels
class_labels = ['Glass', 'Metal', 'Organic', 'Paper', 'Plastic']

def main_page():
    def on_exit():
        if messagebox.askyesno("Exit", "Are you sure you want to quit?"):
            root.destroy()

    def go_to_garbage_types_page():
        root.withdraw()
        garbage_types_page(root)

    root = tk.Tk()
    root.title("Home Page")
    root.geometry("1000x600")

    bg_image = Image.open("D:/Project/OceanWaste.png").convert("RGB")
    bg_image = bg_image.resize((1000, 600))
    bg_photo = ImageTk.PhotoImage(bg_image)

    bg_label = tk.Label(root, image=bg_photo)
    bg_label.image = bg_photo
    bg_label.place(relwidth=1, relheight=1)

    thesis_title = tk.Label(root, text="Garbage Classification using ResNet50", font=("Helvetica", 26))
    thesis_title.place(x=200, y=100)

    start_button = tk.Button(root, text="Start", command=go_to_garbage_types_page, font=("Helvetica", 16), width=10)
    start_button.place(x=350, y=400)

    exit_button = tk.Button(root, text="Exit", command=on_exit, font=("Helvetica", 16), width=10)
    exit_button.place(x=550, y=400)

    presenter_label = tk.Label(root, text="By\nMa Thet Thet Zon", font=("Helvetica", 12), bg="white")
    presenter_label.place(x=800, y=500)

    root.mainloop()

def garbage_types_page(previous_window):
    def go_to_classification_page():
        garbage_window.withdraw()
        classification_page(garbage_window)

    def go_to_augmentation_page():
        garbage_window.withdraw()
        augmentation_page(garbage_window)

    def go_back():
        garbage_window.destroy()
        if previous_window:
            previous_window.deiconify()

    garbage_window = tk.Toplevel()
    garbage_window.title("Garbage Types")
    garbage_window.geometry("1000x600")

    garbage_type_title = tk.Label(garbage_window, text="Garbage Types", font=("Helvetica", 22))
    garbage_type_title.place(x=400, y=20)

    garbages = ["Glass", "Metal", "Paper", "Plastic", "Organic"]
    garbage_images = [
        "D:/Project/RealWaste/Training/Glass/Glass_159.jpg",
        "D:/Project/RealWaste/Training/Metal/Metal_400.jpg",
        "D:/Project/RealWaste/Training/Paper/Paper_428.jpg",
        "D:/Project/RealWaste/Training/Plastic/Plastic_141.jpg",
        "D:/Project/RealWaste/Training/Organic/Organic_2.jpg"
    ]

    img_refs = []

    start_x = 200
    start_y = 75
    x_offset = 180
    y_offset = 220

    for i, garbage in enumerate(garbages):
        img = Image.open(garbage_images[i]).convert("RGB")
        img = img.resize((200, 180))
        img_photo = ImageTk.PhotoImage(img)
        img_refs.append(img_photo)

        x_position = start_x + (i % 3) * x_offset
        y_position = start_y + (i // 3) * y_offset

        img_label = tk.Label(garbage_window, image=img_photo)
        img_label.image = img_photo
        img_label.place(x=x_position, y=y_position)

        lbl = tk.Label(garbage_window, text=garbage, font=("Helvetica", 12))
        lbl.place(x=x_position + 75, y=y_position + 185)

    classify_button = tk.Button(garbage_window, text="Garbage Classification", font=("Helvetica", 12), bg='lightblue', command=go_to_classification_page)
    classify_button.place(x=350, y=530)

    augment_button = tk.Button(garbage_window, text="Image Augmentation", font=("Helvetica", 12), bg='lightblue', command=go_to_augmentation_page)
    augment_button.place(x=550, y=530)

    back_button = tk.Button(garbage_window, text="Back", bg='lightblue', command=go_back)
    back_button.place(x=10, y=0)
    garbage_window.mainloop()

def classification_page(previous_window):
    root = tk.Toplevel()
    root.title("Garbage Classification")
    root.geometry("1000x600")

    def classify_image():
        img_path = input_image_canvas.image_path
        if img_path:
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions)]
            type_of_garbage_display.config(text=predicted_class)
        else:
            messagebox.showwarning("No Image", "Please select an image to classify.")

    def browse_image():
        img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if img_path:
            try:
                img = Image.open(img_path)
                
                # Convert image modes if necessary
                if img.mode in ("P", "RGBA"):
                    img = img.convert("RGBA")
                
                # Resize the image
                img = img.resize((225, 225))
                
                # Convert image to PhotoImage format
                img_tk = ImageTk.PhotoImage(img)
                
                # Clear previous image on canvas
                input_image_canvas.delete("all")
                
                # Display new image
                input_image_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                
                # Store reference to avoid garbage collection
                input_image_canvas.img_tk = img_tk
                input_image_canvas.image_path = img_path

            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image: {e}")

    def clear_image():
        input_image_canvas.delete("all")
        type_of_garbage_display.config(text="")
        input_image_canvas.image_path = None

    def go_back():
        root.destroy()
        if previous_window:
            previous_window.deiconify()

    title_label = tk.Label(root, text="Garbage Classification using ResNet50", font=("Helvetica", 20))
    title_label.place(x=300, y=30)

    input_image_label = tk.Label(root, text="Input Image", font=("Helvetica", 14))
    input_image_label.place(x=460, y=100)

    input_image_canvas = tk.Canvas(root, width=225, height=225, bg="white")
    input_image_canvas.place(x=400, y=140)
    input_image_canvas.image_path = None

    type_of_garbage_label = tk.Label(root, text="Type of Garbage", width=20, font=("Helvetica", 12))
    type_of_garbage_label.place(x=680, y=200)

    type_of_garbage_display = tk.Label(root, text="", font=("Helvetica", 12), bg="white", width=20, height=2)
    type_of_garbage_display.place(x=680, y=240)

    classify_button = tk.Button(root, text="Classify with ResNet-50", width=24, command=classify_image, font=("Helvetica", 13))
    classify_button.place(x=400, y=400)

    browse_button = tk.Button(root, text="Browse Image", command=browse_image, width=15, font=("Helvetica", 12))
    browse_button.place(x=200, y=200)

    clear_button = tk.Button(root, text="Clear", command=clear_image, width=15, font=("Helvetica", 12))
    clear_button.place(x=200, y=250)

    back_button = tk.Button(root, text="Back", command=go_back, font=("Helvetica", 10))
    back_button.place(x=10, y=0)

    root.mainloop()

def augmentation_page(previous_window):
    root = tk.Toplevel()
    root.title("Image Augmentation")
    root.geometry("1000x600")

    def browse_image():
        img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if img_path:
            try:
                img = Image.open(img_path)
                
                # Convert image modes if necessary
                if img.mode in ("P", "RGBA"):
                    img = img.convert("RGBA")
                
                # Resize the image
                img = img.resize((225, 225))
                
                # Convert image to PhotoImage format
                img_tk = ImageTk.PhotoImage(img)
                
                # Clear previous image on canvas
                original_image_canvas.delete("all")
                
                # Display new image
                original_image_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                
                # Store reference to avoid garbage collection
                original_image_canvas.img_tk = img_tk
                original_image_canvas.image_path = img_path

            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image: {e}")

    def apply_augmentation():
        img_path = original_image_canvas.image_path
        if img_path:
            img = Image.open(img_path)
            if img.mode in ("P", "RGBA"):
                img = img.convert("RGBA")

            img = img.resize((150, 150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            augmented_images = datagen.flow(img_array, batch_size=1)
            augmented_imgs = [next(augmented_images)[0] for _ in range(4)]

            for idx, aug_img in enumerate(augmented_imgs):
                aug_img = array_to_img(aug_img)
                aug_img_tk = ImageTk.PhotoImage(aug_img)
                augmented_canvases[idx].delete("all")
                augmented_canvases[idx].create_image(0, 0, anchor=tk.NW, image=aug_img_tk)
                augmented_canvases[idx].img_tk = aug_img_tk

        else:
            messagebox.showwarning("No Image", "Please select an image to augment.")
    def clear_images():
        # Clear the original image and augmented images
        original_image_canvas.delete("all")
        for canvas in augmented_canvases:
            canvas.delete("all")
        
        # Reset image references to avoid garbage collection
        original_image_canvas.img_tk = None
        for canvas in augmented_canvases:
            canvas.img_tk = None

    def go_back():
        root.destroy()
        if previous_window:
            previous_window.deiconify()

    title_label = tk.Label(root, text="Image Augmentation", font=("Helvetica", 20))
    title_label.place(x=400, y=30)

    original_image_label = tk.Label(root, text="Original Image", font=("Helvetica", 12))
    original_image_label.place(x=150, y=100)

    original_image_canvas = tk.Canvas(root, width=200, height=200, bg="white")
    original_image_canvas.place(x=150, y=140)
    original_image_canvas.image_path = None

    augmented_labels = ["Augmented Image 1", "Augmented Image 2", "Augmented Image 3", "Augmented Image 4"]
    augmented_canvases = []
    

    for i, label in enumerate(augmented_labels):
        x_position = 400 + (i % 2) * 250
        y_position = 100 + (i // 2) * 200
        tk.Label(root, text=label, font=("Helvetica", 12)).place(x=x_position, y=y_position)
        canvas = tk.Canvas(root, width=150, height=120, bg="white")
        canvas.place(x=x_position, y=y_position + 40)
        augmented_canvases.append(canvas)

    browse_button = tk.Button(root, text="Browse Image", command=browse_image, width=15, font=("Helvetica", 12))
    browse_button.place(x=180, y=350)

    clear_button = tk.Button(root, text="Clear", command=clear_images, width=15, font=("Helvetica", 12))
    clear_button.place(x=180, y=400)

    augment_button = tk.Button(root, text="Apply Augmentation", command=apply_augmentation, width=20, font=("Helvetica", 12))
    augment_button.place(x=400, y=500)

    back_button = tk.Button(root, text="Back", command=go_back, font=("Helvetica", 10))
    back_button.place(x=10, y=0)

    root.mainloop()

def main():
    main_page()

if __name__ == "__main__":
    main()
