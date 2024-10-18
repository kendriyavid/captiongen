# CaptionCuddler 🖼️✨

CaptionCuddler is a sophisticated web application that generates AI-powered captions for images using a CNN-LSTM neural network model trained on the Flickr8k dataset. The application combines a sleek, minimalist frontend with a robust Flask backend for image processing and caption generation.

## 🌟 Features

- **Advanced AI Model**: CNN-LSTM architecture trained on Flickr8k dataset
- **Clean Dark Interface**: Minimalist design with a professional dark theme
- **Drag & Drop**: Easy image upload with drag and drop functionality
- **Instant Preview**: See your uploaded image before processing
- **Real-time Processing**: Fast and efficient caption generation
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🧠 Model Architecture

Our image captioning system uses a hybrid CNN-LSTM architecture:

- **CNN Component**: VGG16 pre-trained on ImageNet
  - Extracts high-level image features
  - Modified to remove top classification layers
  - Input shape: 224x224x3 RGB images

- **LSTM Component**:
  - Processes image features to generate natural language captions
  - Trained for 40 epochs on Flickr8k dataset
  - Vocabulary size: Based on Flickr8k captions
  - Maximum sequence length: 40 words

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/caption-cuddler.git
cd caption-cuddler
```

2. Set up the Python environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the model files:
```bash
# Create the models directory
mkdir models

# Download the pre-trained models (you'll need to provide these files)
# - models/caption_model.h5
# - models/tokenizer.pkl
# - models/features.pkl (if using pre-extracted features)
```

4. Start the Flask backend:
```bash
python app.py
```

5. Open `index.html` in your web browser

## 💻 Technology Stack

### Frontend
- Pure HTML/CSS
- React (via CDN)
- Vanilla JavaScript

### Backend
- Flask 2.0+
- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy
- Pillow (PIL)
- VGG16 (pre-trained on ImageNet)

## 📦 Requirements

```txt
flask==2.0.1
flask-cors==3.0.10
tensorflow==2.8.0
numpy==1.19.5
Pillow==8.3.1
keras==2.8.0
```

## 🎯 Usage

1. Ensure the Flask backend is running:
```bash
python app.py
```

2. Open the frontend:
   - Open `index.html` in your browser
   - The application will connect to the Flask backend on `http://localhost:5000`

3. Upload an image:
   - Click the upload area or drag an image
   - The image will be preprocessed and sent to the model
   - Receive your AI-generated caption

## 🛠️ API Endpoints

### Image Caption Generation
```
POST /upload
Content-Type: multipart/form-data

Parameters:
- image: file (required) - The image file to generate a caption for

Returns:
{
    "caption": "Generated caption for the image",
    "success": true
}
```

## 📂 Project Structure

```
caption-cuddler/
│
├── app.py              # Flask application with CNN-LSTM model
├── index.html          # Frontend application
├── requirements.txt    # Python dependencies
├── README.md          # Documentation
│
├── models/            # Model files directory
│   ├── caption_model.h5    # Trained LSTM model
│   ├── tokenizer.pkl      # Tokenizer for text processing
│   └── features.pkl      # Pre-extracted image features (optional)
│
└── uploads/           # Temporary directory for uploaded images
```

## 🔧 Model Training Details

- **Dataset**: Flickr8k
- **Training Duration**: 40 epochs
- **Architecture**:
  - VGG16 for image feature extraction
  - LSTM layers for sequence generation
- **Training Parameters**:
  - Batch size: 64
  - Learning rate: 0.001
  - Optimizer: Adam
  - Loss: Categorical Crossentropy
    
## 🤝 Acknowledgments

- Flickr8k dataset for model training
- VGG16 architecture and pre-trained weights
- All contributors and users

---
Made with ❤️ by [Harshdeep Singh]
