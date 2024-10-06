# from flask import Flask, request, jsonify
# from keras.models import load_model
# import numpy as np
# from PIL import Image
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import pickle
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# app = Flask(__name__)

# # Constants
# MAX_LENGTH = 35  # Make sure this matches your training configuration

# # Load the pre-trained model
# model = load_model('./best40modeltf.keras')

# # Load the pre-saved tokenizer
# with open('./tokenizer.pkl', 'rb') as f:
#     tokenizer = pickle.load(f)

# # Create feature extraction model
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(4096, activation='relu')(x)
# feature_extraction_model = Model(inputs=base_model.input, outputs=x)

# def idx_to_word(integer, tokenizer):
#     for word, index in tokenizer.word_index.items():
#         if index == integer:
#             return word
#     return None

# def predict_caption(model, image_features, tokenizer, max_length):
#     in_text = 'startseq'
#     for i in range(max_length):
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
#         yhat = model.predict([image_features, sequence], verbose=0)
#         yhat = np.argmax(yhat)
#         word = idx_to_word(yhat, tokenizer)
#         if word is None:
#             break
#         in_text += " " + word
#         if word == 'endseq':
#             break
#     return in_text

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400

#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     try:
#         # Preprocess the image
#         image = Image.open(file)
#         image = image.resize((224, 224))
#         image = img_to_array(image)
#         image = image / 255.0
#         image = np.expand_dims(image, axis=0)
        
#         # Extract features
#         image_features = feature_extraction_model.predict(image)
        
#         # Generate caption
#         caption = predict_caption(model, image_features, tokenizer, MAX_LENGTH)
        
#         # Clean up the caption
#         caption = caption.replace('startseq', '').replace('endseq', '').strip()
        
#         return jsonify({'caption': caption})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)

# from flask import Flask, request, jsonify
# from keras.models import load_model
# import numpy as np
# from PIL import Image
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import pickle
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# import tensorflow as tf
# from keras.utils import custom_object_scope

# app = Flask(__name__)

# # Constants
# MAX_LENGTH = 35  # Make sure this matches your training configuration

# # Define custom objects
# custom_objects = {
#     'tf': tf,
#     'not_equal': tf.not_equal
# }

# # Load the pre-trained model using custom_object_scope
# print("Loading model...")
# with custom_object_scope(custom_objects):
#     try:
#         tf.keras.backend.clear_session()
#         model = load_model('./best40modeltf.keras')
#         print("Model loaded successfully!")
#     except Exception as e:
#         print(f"Error loading model: {str(e)}")
#         raise

# # Load the pre-saved tokenizer
# print("Loading tokenizer...")
# with open('./tokenijer.pkl', 'rb') as f:
#     tokenizer = pickle.load(f)
# print("Tokenizer loaded successfully!")

# # Create feature extraction model
# print("Setting up feature extraction model...")
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(4096, activation='relu')(x)
# feature_extraction_model = Model(inputs=base_model.input, outputs=x)
# print("Feature extraction model ready!")

# def preprocess_image(image_path):
#     # Load and preprocess the image
#     if isinstance(image_path, str):
#         img = load_img(image_path, target_size=(224, 224))
#     else:
#         img = Image.open(image_path).convert('RGB')
#         img = img.resize((224, 224))
    
#     # Convert to array and preprocess
#     x = img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)  # VGG16 preprocessing
#     return x

# def idx_to_word(integer, tokenizer):
#     for word, index in tokenizer.word_index.items():
#         if index == integer:
#             return word
#     return None

# def predict_caption(model, image_features, tokenizer, max_length):
#     try:
#         # Initialize the input sequence
#         in_text = 'startseq'
        
#         # Generate caption word by word
#         for _ in range(max_length):
#             # Encode the current input sequence
#             sequence = tokenizer.texts_to_sequences([in_text])[0]
            
#             # Pad the sequence
#             sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
            
#             # Predict next word
#             with tf.device('/CPU:0'):  # Force CPU prediction to avoid potential GPU memory issues
#                 yhat = model.predict([image_features, sequence], verbose=0)
            
#             # Get the word with highest probability
#             word_idx = np.argmax(yhat)
            
#             # Convert the index to a word
#             word = idx_to_word(word_idx, tokenizer)
            
#             # Stop if we can't find the word or reach the end token
#             if word is None or word == 'endseq':
#                 break
                
#             # Append the word to the sequence
#             in_text += ' ' + word
            
#         return in_text
#     except Exception as e:
#         print(f"Error in predict_caption: {str(e)}")
#         raise

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     try:
#         # Preprocess the image
#         print("Preprocessing image...")
#         image = preprocess_image(file)
        
#         # Extract features
#         print("Extracting features...")
#         image_features = feature_extraction_model.predict(image, verbose=0)
#         print(f"Feature shape: {image_features.shape}")
        
#         # Generate caption
#         print("Generating caption...")
#         caption = predict_caption(model, image_features, tokenizer, MAX_LENGTH)
#         print(f"Raw caption: {caption}")
        
#         # Clean up the caption
#         caption = caption.replace('startseq', '').replace('endseq', '').strip()
#         print(f"Final caption: {caption}")
        
#         return jsonify({'caption': caption})
    
#     except Exception as e:
#         print(f"Error in prediction: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)


from flask import Flask, request, jsonify
from keras.models import load_model
from flask_cors import CORS  # Import CORS
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}})
# Constants
MAX_LENGTH = 35

# Load the main caption generation model
print("Loading caption model...")
model = tf.keras.models.load_model('./best40modeltf.keras')
print("Caption model loaded successfully!")

# Load the tokenizer
print("Loading tokenizer...")
with open('./tokenijer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded successfully!")

# Create feature extraction model exactly as in Kaggle notebook
print("Setting up feature extraction model...")
vgg_model = VGG16()
feature_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
print("Feature extraction model ready!")

def extract_features(image):
    """Extract features using VGG16 model"""
    try:
        # Ensure the image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to VGG16 required size
        image = image.resize((224, 224))
        
        # Convert image to array
        image = img_to_array(image)
        
        # Reshape for model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        
        # Preprocess image for VGG16
        image = preprocess_input(image)
        
        # Extract features
        features = feature_model.predict(image, verbose=0)
        
        return features
    
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        raise

def idx_to_word(integer, tokenizer):
    """Convert word index to word"""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image_features, tokenizer, max_length):
    """Generate caption for the image"""
    try:
        in_text = 'startseq'
        print(f"Initial input text: {in_text}")
        
        for i in range(max_length):
            # Convert the current text to sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            
            # Pad the sequence
            sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
            print(f"Sequence for prediction: {sequence}")
            
            # Predict next word
            yhat = model.predict([image_features, sequence], verbose=0)
            yhat = np.argmax(yhat)
            print(f"Predicted index: {yhat}")
            
            # Check if the predicted index is valid
            if yhat >= len(tokenizer.word_index) + 1:
                break
            
            # Convert index to word
            word = idx_to_word(yhat, tokenizer)
            
            # Break if we can't find the word or reach the end
            if word is None or word == 'endseq':
                break
            
            # Append the word
            in_text += " " + word
            print(f"Current caption: {in_text}")
        
        return in_text
    
    except Exception as e:
        print(f"Error in caption prediction: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image caption prediction requests"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Open and process the image
        print("\nProcessing image...")
        image = Image.open(file)
        
        # Extract features
        print("Extracting features...")
        image_features = extract_features(image)
        print(f"Feature shape: {image_features.shape}")
        print(f"Feature stats - Min: {image_features.min():.4f}, Max: {image_features.max():.4f}, Mean: {image_features.mean():.4f}")
        
        # Generate caption
        print("Generating caption...")
        caption = predict_caption(model, image_features, tokenizer, MAX_LENGTH)
        print(f"Raw caption: {caption}")
        
        # Clean up the caption
        final_caption = caption.replace('startseq', '').replace('endseq', '').strip()
        print(f"Final caption: {final_caption}")
        
        return jsonify({'caption': final_caption})
    
    except Exception as e:
        print(f"Error in prediction pipeline: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)