import os
import shutil

from classifier.predictor import Predictor


class BatchClassifier:

    def __init__(self):

        print("Loading Predictor...")

        self.predictor = Predictor()

    def classify_folder(

        self,

        input_folder,

        output_folder

    ):

        image_extensions = (

            ".png",

            ".jpg",

            ".jpeg",

            ".bmp",

            ".tif",

            ".tiff"

        )

        total = 0

        for file in os.listdir(input_folder):

            if file.lower().endswith(image_extensions):

                total += 1

        print(f"\nFound {total} images\n")

        count = 0

        for file in os.listdir(input_folder):

            if not file.lower().endswith(image_extensions):

                continue

            image_path = os.path.join(

                input_folder,

                file

            )

            result = self.predictor.predict(

                image_path

            )

            prediction = result["prediction"]
            confidence = result["confidence"]

            destination_folder = os.path.join(

                output_folder,

                prediction

            )

            os.makedirs(

                destination_folder,

                exist_ok=True

            )

            shutil.move(

                image_path,

                os.path.join(

                    destination_folder,

                    file

                )

            )

            count += 1

            print(

                f"[{count}/{total}]",

                file,

                "->",

                prediction,

                f"({confidence:.2f}%)"

            )

        print("\nFinished.")