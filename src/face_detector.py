import os
import cv2
from mtcnn import MTCNN

# initialize the mtcnn face detector - it detects human faces and returns bounding box coordinates
detector = MTCNN()


def process_and_crop_faces(input_base_path, output_base_path, target_size=(48, 48)):

    # get all emotion category folders from dataset
    emotions = [
        d for d in os.listdir(input_base_path)
        if os.path.isdir(os.path.join(input_base_path, d))
    ]

    for emotion in emotions:

        # create full input path for current emotion
        input_dir = os.path.join(input_base_path, emotion)
        # create full output path for processed images
        output_dir = os.path.join(output_base_path, emotion)

        os.makedirs(output_dir, exist_ok=True)

        # iterate through all images inside emotion folder
        for filename in os.listdir(input_dir):

            img_path = os.path.join(input_dir, filename)

            # load image using opencv
            img = cv2.imread(img_path)

            if img is None:
                print(f"Could not read image: {filename}")
                continue

            # convert image from bgr to rgb
            # opencv loads images as bgr by default
            # mtcnn expects rgb format
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # detect faces in current image
            # detector returns list of detected faces
            faces = detector.detect_faces(img_rgb)

            if faces:

                # select first detected face
                # assumption: image contains one main person
                x, y, w, h = faces[0]['box']

                # ensure coordinates are not negative
                # prevents indexing errors during cropping
                x = max(0, x)
                y = max(0, y)

                # crop face region from original image
                cropped_face = img[y:y+h, x:x+w]

                # resize cropped face to 48x48 pixels (fer-2013 image dimensions)
                resized_face = cv2.resize(cropped_face, target_size)

                # create output path for processed image
                output_path = os.path.join(output_dir, filename)

                # save processed face image
                cv2.imwrite(output_path, resized_face)

                print(f"Successfully processed: {filename}")

            else:
                print(f"No face detected in: {filename}")


if __name__ == "__main__":

    print("Starting face detection and cropping module...")

    input_dataset = "../data/custom_dataset"
    output_dataset = "../data/processed/custom_dataset"

    if os.path.exists(input_dataset):

        process_and_crop_faces(input_dataset, output_dataset)

        print("Processing complete. Check the output folder.")

    else:
        print(f"Error: Directory '{input_dataset}' not found. Please ensure it exists.")