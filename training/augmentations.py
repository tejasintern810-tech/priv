from torchvision import transforms


############################################################
# ImageNet Statistics
############################################################

MEAN = [0.485, 0.456, 0.406]

STD = [0.229, 0.224, 0.225]


############################################################
# Training Transform
############################################################

train_transform = transforms.Compose(

    [

        transforms.Resize((224, 224)),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(8),

        transforms.ColorJitter(

            brightness=0.20,

            contrast=0.20,

            saturation=0.10,

            hue=0.03

        ),

        transforms.RandomAffine(

            degrees=0,

            translate=(0.05, 0.05),

            scale=(0.95, 1.05)

        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=MEAN,

            std=STD

        )

    ]

)


############################################################
# Validation Transform
############################################################

val_transform = transforms.Compose(

    [

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=MEAN,

            std=STD

        )

    ]

)