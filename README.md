# Image-Based-Open-Circuit-Fault-Diagnosis-in-Power-Inverters
Deep learning-based framework for open-circuit fault diagnosis in three-phase voltage source inverters using α–β current trajectory imaging and ResNet-50 / Vision Transformer models.

This repository presents a deep learning-based framework for detecting and classifying open-circuit faults in three-phase Voltage Source Inverters (VSIs) feeding induction motors.

The proposed approach transforms three-phase stator currents into the α–β reference frame using the Clarke–Concordia transformation, and encodes the resulting trajectories as grayscale images. These images are then used to train and evaluate deep learning models, including ResNet-50 and Vision Transformer (ViT), for multi-class fault classification.
