import cv2
import os

stitcher = cv2.Stitcher_create()

def stitch_images(image_folder):
    images = []
    
    for filename in sorted(os.listdir(image_folder)):  # Ensure correct order
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(image_folder, filename))
            images.append(img)

    if len(images) < 2:
        print("Need at least two images to stitch.")
        return None
    
    status, result = stitcher.stitch(images)
    
    if status == cv2.STITCHER_OK:
        print("Stitching Successful!")
        return result
    else:
        print(f"Stitching Failed. Error Code: {status}")
        return None

image_folder = "Images/"

stitched_image = stitch_images(image_folder)

if stitched_image is not None:
    cv2.imshow(f"Stitched Image : {stitched_image}")
    cv2.imwrite(f"stitched_output.jpg: {stitched_image}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
