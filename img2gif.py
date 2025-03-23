import cv2
import os, os.path as osp
import imageio

def images_to_gif(image_folder, output_path, prefix="output_test_", ext=".png", fps=10):
    images = []
    for i in range(1, 161):
        img_path = os.path.join(image_folder, f"{prefix}{i}{ext}")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
            # Convert from BGR (OpenCV default) to RGB for correct GIF colors
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        else:
            print(f"Warning: {img_path} not found")

    if not images:
        print("No images found.")
        return

    # Calculate frame duration in seconds (duration per frame = 1 / fps)
    duration = 1.0 / fps
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, images, duration=duration)
    print(f"GIF saved to {output_path}")

# Create GIFs from images with different prefixes.
images_to_gif(
    image_folder="output/images",
    output_path="assets/output_test_video.gif",
    prefix="output_test_",
    ext=".png",
    fps=20
)

images_to_gif(
    image_folder="output/images",
    output_path="assets/output_val_video.gif",
    prefix="output_val_",
    ext=".png",
    fps=20
)
