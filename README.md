Garbage Classification Using ResNet50


This repository contains the code and resources for a garbage classification project using a fine-tuned ResNet50 model. The project aims to classify different types of waste, such as glass, metal, paper, plastic, and organic waste, using a pretrained deep learning model.

Project Overview

This project leverages the ResNet50 model, a powerful deep learning model pretrained on the ImageNet dataset, to classify images of waste into five categories: glass, metal, paper, plastic, and organic. By fine-tuning the model on the RealWaste dataset, the model can accurately identify different types of garbage.

The repository contains:

garbageclassification.ipynb: A Jupyter Notebook detailing the model training and fine-tuning process.
GUI_for_GC.py: The code for a graphical user interface (GUI) that allows users to classify images of garbage easily.

Dataset

The RealWaste dataset was used for training and testing the model. It includes a diverse set of images for each waste category, making it suitable for building a robust classification model.

Model Architecture

The ResNet50 model, a deep convolutional neural network with 50 layers, is used as the base model. The model was pretrained on the ImageNet dataset, which allowed us to leverage transfer learning for this task. After fine-tuning the model on the RealWaste dataset, we achieved improved performance for the specific task of garbage classification.

Fine-Tuning

The fine-tuning process involved unfreezing some of the top layers of the ResNet50 model and training them on the RealWaste dataset. This helped the model learn more specific features related to garbage classification while retaining the general features learned from ImageNet.
