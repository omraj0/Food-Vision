# 🍲 Food Image Classification

This project demonstrates an image classification application using TensorFlow and Streamlit. The model is trained to classify images of various Indian food items.

## 📋 Table of Contents
- [📦 Installation](#-installation)
- [🚀 Usage](#-usage)
- [🔍 Model Training](#-model-training)
  - [ResNet50](#resnet50)
  - [EfficientNet](#efficientnet)
- [🤖 Model Prediction](#-model-prediction)
- [📁 Project Structure](#-project-structure)
- [🙏 Acknowledgements](#-acknowledgements)

## 📦 Installation
### Clone the repository:
   ```bash
   git clone https://github.com/your-username/food-image-classification.git
   cd food-image-classification
  ```
**Install the required packages:**
```bash
pip install -r requirements.txt
```
## 🚀 Usage
To run the Streamlit app locally, use the following command:
```bash
streamlit run app.py
```
## 🔍 Model Training

The model is based on two popular architectures: ResNet50 and EfficientNetV2B0, both of which are trained on a dataset of Indian food images.

### ResNet50

ResNet50 is a deep residual network that helps to mitigate the vanishing gradient problem by using residual blocks. This model is highly effective for image classification tasks and has been pre-trained on the ImageNet dataset.

### EfficientNet

EfficientNetV2B0 is a family of models that efficiently scale up the size of the network. It balances network depth, width, and resolution to achieve better performance with fewer parameters.

### Training Process

- **Data Preparation**: The images are loaded and preprocessed using TensorFlow's `tf.keras.preprocessing.image_dataset_from_directory` method.
- **Model Architecture**: The model uses either ResNet50 or EfficientNetV2B0 as the base model with a custom top layer for classification into 80 classes.
- **Training**: The model is trained with data augmentation to improve its generalization capability. The training script can be found in the `train.py` file.

## 🤖 Model Prediction

The prediction script (`img_classification.py`) loads a trained model and makes predictions on new images. The `load_and_prep_image` function preprocesses the image, and the `predict_data` function makes the prediction and returns the predicted class.

## 📁 Project Structure

```bash
.
├── app.py                      # Streamlit app script
├── img_classification.py       # Image classification script
├── train.py                    # Model training script
├── requirements.txt            # Python packages required
└── README.md                   # Project README file
```
## 🙏 Acknowledgements
This project is inspired by the Food-101 dataset and uses the ResNet50 and EfficientNetV2B0 models from TensorFlow.
