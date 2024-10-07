# Coronary Stenosis Automatic Detection with YOLO

Welcome to the **Coronary Stenosis Detection Project**! This project is a playful experiment in machine learning, driven by a cardiologist's curiosity (that's me!) to explore how computer vision can be applied in coronary imaging. Please remember, I'm an interventional cardiologist—not a computer vision expert. So, this is just for fun, learning, and maybe a little bit of usefulness!

## Overview
This project leverages the **YOLOv11n-seg** model, trained on a dataset of coronary angiograms, to automatically detect and segment areas of stenosis in coronary arteries. It's built using **Python** and **Streamlit** to create an interactive web interface.

## Project Inspiration
I've always been fascinated by the potential of AI to assist in healthcare, particularly in my field of cardiology. With the right tools, we can gain valuable insights and even improve diagnostic efficiency. With a nudge from curiosity (and a lot of help from GPT o1), I set out to experiment and see what we could do with automatic stenosis detection. 

## About the Project
1. **Data:**  
   I used the **ARCADE Coronary Angiography Dataset** (annotated dataset available on [Zenodo](https://zenodo.org/records/8386059)). This dataset is specifically tailored to coronary angiograms and was invaluable for training the model.
   
2. **Model Training:**  
   I trained a **YOLOv11n-seg** model using **Ultralytics HUB**. The process was quite straightforward and took around **3 hours** on a free-tier account. The goal was to allow the model to both detect and segment areas of stenosis.

3. **App Development:**  
   With the assistance of **GPT-o1**, I developed an interactive app using **Streamlit**. This app allows users to upload their own coronary angiograms or try some sample images. It will then attempt to identify and segment potential areas of stenosis. 

## Technical Details & Performance
Here are the model's evaluation metrics:
- **mAP@0.5**: 0.377
- **Precision**: 0.458

While these numbers might not blow you away, let's keep in mind:
- This model handles **segmentation** with numerous instances, which is quite a challenging task.
- This was a quick experiment, more for entertainment and education than to replace a radiologist just yet. 

## How to Run the App
1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/yourrepository.git
    cd yourrepository
    ```

2. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the app**:
    ```bash
    streamlit run app.py
    ```

4. **Upload an Image or Choose a Sample**:  
   Use the app to upload your own coronary angiogram image or select one of the provided sample images to see the model in action.

## Future Improvements
- **Better Metrics**: As mentioned, this is more of an experiment. With some tweaks and better data, it would be exciting to see what kind of metrics we could achieve.
- **Model Optimization**: Potential to experiment with other architectures or a custom training routine.
- **User Interface**: Streamlining the user experience, especially for non-technical users.

Thanks for stopping by, and if you’re another cardiologist with an interest in AI, I hope you find this as interesting as I did!

---
_Disclaimer: This project is intended for experimentation and educational purposes only and should not be used for medical diagnosis or treatment decisions._
