# Campus Vision Challenge Model

This README provides step-by-step instructions on how to test the model for the Campus Vision Challenge.

# team se7en
 - Kofi Acheampong Ennin
 - Sandys Ayuumah

## Requirements

- Python 3.7+
- CUDA-capable GPU (recommended)
- See `requirements.txt` for required packages

## Setup

1. Clone this repository:


2. Install required packages:


3. Download the model weights:
- Download `best_model.pth` from https://drive.google.com/drive/folders/1HUfknjhYScrgF9uFHLlaMTdBRALDI8C-?usp=sharing
- Place `best_model.pth` in the root directory of the project

## Testing the Model

1. Place your test images in a directory named `pred_folder`

2. Run the `infer.py` with internet avaliable. 

3. `infer.py` will output the predictions in your terminal 


## Additional Information

- The model is based on a RESNET50 pretrained on imagenet from pytorch
- Before choosing this architecture, we tried efficientnet, but Resnet proved better for the task.
- compute limit using Colab may not have allowed as to explore full capabilities.

# the result of the model after training for 11 epochs is as follows:

    Accuracy: 0.9945
    Precision: 0.9945
    Recall: 0.9945
    F1 Score: 0.9945
    Log Loss: 0.0300

    loss graph
    [alt text](https://photos.app.goo.gl/177GLfPV5MRRg6TL7)