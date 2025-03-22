import cv2
import os

def images_to_video(image_folder, output_path, prefix="output_test_", ext=".png", fps=10):
    images = []
    for i in range(1, 161):
        img_path = os.path.join(image_folder, f"{prefix}{i}{ext}")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            images.append(img)
        else:
            print(f"Warning: {img_path} not found")

    if not images:
        print("No images found.")
        return

    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img in images:
        video.write(img)

    video.release()
    print(f"Video saved to {output_path}")

images_to_video(
    image_folder="output/images",
    output_path="output_test_video.mp4",
    prefix="output_test_",
    ext=".png",
    fps=10
)

images_to_video(
    image_folder="output/images",
    output_path="output_val_video.mp4",
    prefix="output_val_",
    ext=".png",
    fps=10
)
