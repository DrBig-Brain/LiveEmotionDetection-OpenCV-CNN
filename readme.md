# Live Facial Emotion Detection with OpenCV and CNN

A deep learning project that performs real-time facial emotion recognition using Convolutional Neural Networks (CNN) and OpenCV.

## Features

- Real-time emotion detection through webcam
- Support for 7 different emotions
- Pre-trained model included
- Easy to use interface
- GPU acceleration support
- Detailed visualization of results
- Web interface using Streamlit

## Overview

This project implements a facial emotion recognition system that can detect 7 different emotions from facial expressions:
1. Angry 😠
2. Disgust 🤢
3. Fear 😨
4. Happy 😊
5. Neutral 😐
6. Sad 😢
7. Surprise 😲

## Project Structure

```
├── archive/
│   ├── test/              # Test image dataset
│   ├── train/            # Training image dataset
│   ├── test_labels.csv    # Labels for test images
│   └── train_labels.csv   # Labels for training images
├── cnn.keras              # Saved trained model
├── FacialEmotionRecognition.ipynb  # Main training notebook
├── objectDetection.py     # Real-time detection script
├── app.py                # Streamlit web interface
├── requirements.txt       # Project dependencies
└── README.md
```

## Requirements

### Hardware Requirements
- CPU: Intel i5/AMD Ryzen 5 or better
- RAM: 8GB minimum (16GB recommended)
- GPU: NVIDIA GPU with CUDA support (optional but recommended)
- Webcam: 720p or higher resolution

### Software Requirements
- Windows 10/11 or Linux
- Python 3.8 or higher
- CUDA Toolkit 11.2+ (for GPU acceleration)
- Dependencies:
  - TensorFlow >= 2.8.0
  - OpenCV >= 4.5.0
  - NumPy >= 1.19.0
  - Pandas >= 1.3.0
  - Matplotlib >= 3.3.0
  - Streamlit >= 1.24.0

## Quick Start

1. Clone this repository:
```bash
git clone <repository-url>
cd LiveEmotionDetection-OpenCV-CNN
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Choose your preferred interface:

   A. Command line interface:
   ```bash
   python objectDetection.py
   ```

   B. Web interface:
   ```bash
   streamlit run app.py
   ```

## Model Architecture

The CNN model consists of:
- Multiple convolutional layers for feature extraction
- MaxPooling layers for dimension reduction
- Dense layers for classification
- Batch normalization layers for training stability
- Final softmax layer for 7-class emotion prediction

## Training

The model was trained on the RAF-DB dataset with:
- Image size: 100x100 pixels
- Batch size: 32
- Optimizer: RMSprop
- Loss function: Categorical crossentropy
- Epochs: 100
- Data augmentation applied (rotation, shift, zoom, brightness)

## Performance

The model achieved:
- Training accuracy: 91.12%
- Validation accuracy: 80.22%
- Training loss: 0.2602
- Validation loss: 0.7263

## Usage

### Training Mode
1. Open `FacialEmotionRecognition.ipynb` in Jupyter Notebook
2. Follow the step-by-step instructions
3. Run all cells to train the model

### Command Line Interface
1. Ensure webcam is connected
2. Run:
```bash
python objectDetection.py
```
3. Press 'q' to quit the application

### Web Interface
1. Ensure webcam is connected
2. Launch the Streamlit app:
```bash
streamlit run app.py
```
3. Use the sidebar controls to adjust settings
4. Click Start/Stop to control the camera feed

## Troubleshooting

Common issues and solutions:

1. **ImportError: No module named 'tensorflow'**
   ```bash
   pip install tensorflow==2.8.0
   ```

2. **Webcam not detected**
   - Check webcam connections
   - Try different camera index:
   ```python
   cv2.VideoCapture(1)  # Change from 0 to 1 or 2
   ```

3. **CUDA/GPU errors**
   - Verify NVIDIA drivers
   - Check CUDA compatibility
   - Use CPU-only mode if needed

4. **Streamlit Interface Issues**
   - Check port availability
   - Verify webcam permissions
   - Restart the application

## Dataset

The project uses the RAF-DB (Real-world Affective Faces Database) dataset:
- 12,271 training images
- 3,068 test images
- Images aligned and labeled with 7 emotion categories
- High-quality facial expressions
- Diverse demographic representation

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

This project is for educational purposes only. Not for commercial use.

## Acknowledgments

- RAF-DB dataset creators
- SPEC, NITH
- OpenCV community
- TensorFlow team
- Streamlit community

## Contact

For questions and support, please open an issue in