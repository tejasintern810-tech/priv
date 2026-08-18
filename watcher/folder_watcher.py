import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import config
from classifier.predictor import Predictor


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
}


class FolderWatcher:

    def __init__(self):

        self.predictor = Predictor()

        self.executor = ThreadPoolExecutor(
            max_workers=config.MAX_WORKERS
        )

        self.processing_files = set()

    ##############################################################

    def process_image(self, image_path):

        filename = os.path.basename(image_path)

        print("\n----------------------------")
        print(f"Processing : {filename}")

        move_start = time.perf_counter()

        try:

            result = self.predictor.predict(image_path)

            prediction = result["prediction"]
            confidence = result["confidence"]
            timings = result["timings"]

            print(f"Prediction : {prediction}")
            print(f"Confidence : {confidence:.2f}%")

            destination_folder = os.path.join(
                config.OUTPUT_FOLDER,
                prediction
            )

            os.makedirs(
                destination_folder,
                exist_ok=True
            )

            shutil.move(
                image_path,
                os.path.join(destination_folder, filename)
            )

            move_time = (
                time.perf_counter() - move_start
            ) * 1000

            print("\nTiming")
            print(f"Feature Extraction : {timings['feature_extraction']:.2f} ms")
            print(f"FAISS Search       : {timings['faiss']:.2f} ms")
            print(f"Metadata           : {timings['metadata']:.2f} ms")
            print(f"Voting             : {timings['vote']:.2f} ms")
            print(f"Complete           : {timings['total']:.2f} ms")
            print(f"Move File          : {move_time:.2f} ms")
            print("Moved Successfully")

        except Exception as e:

            print(f"Error processing {filename}")
            print(e)

        finally:

            self.processing_files.discard(filename)

    ##############################################################

    def run(self):

        print("\nWatching input folder...")

        while True:

            try:

                with os.scandir(config.INPUT_FOLDER) as entries:

                    for entry in entries:

                        if not entry.is_file():
                            continue

                        filename = entry.name

                        if filename in self.processing_files:
                            continue

                        _, extension = os.path.splitext(filename)

                        if extension.lower() not in IMAGE_EXTENSIONS:
                            continue

                        self.processing_files.add(filename)

                        self.executor.submit(
                            self.process_image,
                            entry.path
                        )

                time.sleep(config.WATCH_DELAY)

            except KeyboardInterrupt:

                print("\nStopped.")
                break

            except Exception as e:

                print("\nError :", e)
                time.sleep(1)