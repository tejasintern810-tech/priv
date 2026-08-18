from classifier.batch_classifier import BatchClassifier


classifier = BatchClassifier()

classifier.classify_folder(

    input_folder="input",

    output_folder="output"

)