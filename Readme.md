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
    ![Loss Graph](https://private-user-images.githubusercontent.com/124000529/384628458-07ce4af0-2af0-487d-881e-19822ca88230.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzExNzQzNjYsIm5iZiI6MTczMTE3NDA2NiwicGF0aCI6Ii8xMjQwMDA1MjkvMzg0NjI4NDU4LTA3Y2U0YWYwLTJhZjAtNDg3ZC04ODFlLTE5ODIyY2E4ODIzMC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMTA5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTEwOVQxNzQxMDZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1kMjA4MTZmYWFiYmI2MWYxOGZjY2RmNTk3MzFjNDI3NjZmMWJjOTljZmE1MDdiYmJkZmUzOTMxYTViNTFhMGYyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.JC2GcbyqeMWrdhhu6pd1WBDZwISzkpNpVNTK6Bsd-y8)