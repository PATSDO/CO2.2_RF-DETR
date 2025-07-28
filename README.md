**RF-DETR for Airport Airplane Detection**
Overview

This project implements the RF-DETR (Robust Faster DETR) model for detecting airplanes in airport environments. The model is trained on a custom dataset downloaded from Roboflow and evaluated using various metrics, including Average Precision (AP), Average Recall (AR), and Mean Average Precision (mAP). 

**Features**
- Training RF-DETR on a custom airport airplane detection dataset.
- Visualization of training and validation loss over epochs.
- Evaluation of model performance using AP, AR, and mAP metrics.
- Inference on test images with bounding box annotations.
- Confusion matrix generation for detailed performance analysis.

**Requirements**
- Python 3.x
- NVIDIA GPU (for training)

**  Required libraries:**
    - rfdetr
    - roboflow
    - supervision
    - matplotlib
    - pandas
    - Pillow
    - tqdm

**Installation**
Install the required dependencies:
    
    pip install rfdetr roboflow supervision matplotlib pandas Pillow tqdm

Ensure you have an NVIDIA GPU and the appropriate drivers installed. Verify GPU availability with:
bash

    nvidia-smi

**Usage**

Download the Dataset:
- The dataset is downloaded using the Roboflow API. Replace the api_key with your own key if necessary.

Train the Model:
- The model is trained for 20 epochs with a batch size of 4 and gradient accumulation steps of 4. Adjust these parameters as needed.

      model = RFDETRBase(epochs=20)
      model.train(dataset_dir=dataset.location, epochs=20, batch_size=4, grad_accum_steps=4, lr=1e-4)

Visualize Training Metrics:
- Plot the training and validation loss over epochs:

      plt.plot(df['epoch'], df['train_loss'], label='Training Loss')
      plt.plot(df['epoch'], df['test_loss'], label='Validation Loss')
      plt.show()

Evaluate Model Performance:
- Compute and visualize AP and AR:

      plt.plot(df['epoch'], df['avg_precision'], label='AP')
      plt.plot(df['epoch'], df['avg_recall'], label='AR')
      plt.show()

Run Inference:
- Perform inference on test images and visualize the results:

      detections = model.predict(image, threshold=0.5)
      sv.plot_images_grid(images=[annotation_image, detections_image], grid_size=(1,2))

Generate Confusion Matrix:
- Evaluate the model's performance across classes:

      confusion_matrix = sv.ConfusionMatrix.from_detections(predictions, targets, classes=ds.classes)
      plt.show()

Notes
- Ensure the dataset path (dataset.location) is correctly set after downloading.
- Adjust hyperparameters (e.g., batch_size, lr) based on your hardware and dataset size.
- For large datasets, consider increasing the number of epochs or using a smaller batch size to fit GPU memory.
