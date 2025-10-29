# 🎯 Object Detection with DETR (Facebook ResNet-50)

This project demonstrates how to perform **object detection** on images using the **Hugging Face Transformers** library and the **DETR (DEtection TRansformer)** model by Facebook.  
It detects and labels multiple objects within an image, draws bounding boxes, and saves the annotated output.

---

## 🚀 Features
- Uses **Hugging Face Transformers** for object detection (`facebook/detr-resnet-50`)
- Automatically detects and labels multiple objects in an image
- Draws bounding boxes around detected objects
- Allows easy **customization** via your own `image_labeling.py` module
- Supports image resizing and saving labeled outputs

---

## 🧾 Results

Below are the results of running **object detection** using the `facebook/detr-resnet-50` model from **Hugging Face Transformers**.  
Each object in the image is detected, labeled, and annotated with a bounding box and confidence score.

---

### 🎯 Example — Object Detection on Image

| 🖼️ Labeled Output | ✅ Original Image |
|------------------|------------------|
| <img width="1004" height="576" alt="Original Image" src="https://github.com/user-attachments/assets/f82f0558-dc7f-4ee9-9731-b8a19b6e2215" /> | <img width="1463" height="902" alt="Labeled Image" src="https://github.com/user-attachments/assets/c184a20d-d59f-40f2-8beb-c3a97652ecb6" /> |

---


