import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os
import pandas as pd
from datetime import datetime
import io
import zipfile # Import for handling ZIP files
import shutil # Import for file operations

# --- CONFIGURATION ---

# Set page config
st.set_page_config(
    page_title="Groundnut Tikka Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
MODEL_FILE = 'tikka_disease_model.h5'
INDICES_FILE = 'class_indices.npy'
TEMP_DIR = "temp_training_data" # Directory to extract ZIP files

# Ensure SEVERITY_LEVELS keys match the expected output indices (0, 1, 2, 3)
SEVERITY_LEVELS = {
    0: "Healthy",
    1: "Mild Tikka Disease",
    2: "Moderate Tikka Disease", 
    3: "Severe Tikka Disease"
}
SEVERITY_COLORS = {
    0: "green",
    1: "yellow",
    2: "orange",
    3: "red"
}

# --- MODEL FUNCTIONS ---

# Model building function
def create_model(num_classes=4):
    """Create a transfer learning model using MobileNetV2"""
    with st.spinner('Loading base model weights...'):
        base_model = MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Training function
@st.cache_resource
def train_model(training_path, epochs=10):
    """Train the model on provided dataset"""
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
        zoom_range=0.2,
        shear_range=0.2
    )
    
    try:
        train_generator = train_datagen.flow_from_directory(
            training_path,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )
    except Exception as e:
        # Re-raise with context for debugging
        raise FileNotFoundError(f"Error loading training data. Check the path and folder structure: {e}")

    if train_generator.samples == 0:
        raise ValueError("No training images found in the specified path. Check subdirectories/unzipped structure.")

    val_generator = train_datagen.flow_from_directory(
        training_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    num_classes = len(train_generator.class_indices)
    model = create_model(num_classes)
    
    # Training with progress
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )
    
    return model, history, train_generator.class_indices

# Prediction function
def predict_image(model, image, class_indices):
    """Predict disease severity for a single image"""
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)
    
    if predictions.shape[1] != len(SEVERITY_LEVELS):
        st.warning(f"Model trained with {predictions.shape[1]} classes. Expected {len(SEVERITY_LEVELS)}. Results might be mapped incorrectly.")

    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index] * 100
    
    severity_name = SEVERITY_LEVELS.get(predicted_class_index, f"Unknown Class Index {predicted_class_index}")
    
    return predicted_class_index, confidence, severity_name, predictions[0]

# Save/Load model functions
def save_model(model, class_indices):
    """Save model and class indices using cross-platform paths"""
    model.save(os.path.join(MODEL_FILE))
    np.save(os.path.join(INDICES_FILE), class_indices)
    
@st.cache_resource
def load_model():
    """Load saved model and class indices using cross-platform paths"""
    model_path = os.path.join(MODEL_FILE)
    indices_path = os.path.join(INDICES_FILE)

    if os.path.exists(model_path) and os.path.exists(indices_path):
        try:
            model = keras.models.load_model(model_path)
            class_indices = np.load(indices_path, allow_pickle=True).item()
            return model, class_indices
        except Exception as e:
            st.error(f"Failed to load model or indices: {e}")
            return None, None
    return None, None

# --- NEW FUNCTION FOR ZIP UPLOAD HANDLER ---

def unzip_and_train(uploaded_zip_file, epochs):
    """Handles the extraction of ZIP data and initiates model training."""
    
    # 1. Clean up old temporary directory if it exists
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 2. Save the uploaded file to a temporary disk location
    zip_path = os.path.join(TEMP_DIR, uploaded_zip_file.name)
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip_file.getbuffer())

    # 3. Extract the contents
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TEMP_DIR)
        
        # 4. Find the actual training directory inside the extracted files
        # Assumption: The ZIP contains a single root folder (e.g., training_data/)
        extracted_contents = os.listdir(TEMP_DIR)
        
        # Look for the directory that contains the class folders (healthy, mild, etc.)
        training_root = TEMP_DIR
        if len(extracted_contents) == 1 and os.path.isdir(os.path.join(TEMP_DIR, extracted_contents[0])):
            # If the zip extracted into a single folder (e.g., temp_training_data/training_data/)
            training_root = os.path.join(TEMP_DIR, extracted_contents[0])

        # 5. Start Training
        model, history, class_indices = train_model(training_root, epochs)
        
        # 6. Clean up temporary files after successful training
        shutil.rmtree(TEMP_DIR)
        
        return model, history, class_indices

    except zipfile.BadZipFile:
        st.error("❌ Invalid ZIP file. Please ensure the file is not corrupted.")
        shutil.rmtree(TEMP_DIR)
        return None, None, None
    except Exception as e:
        st.error(f"❌ Training setup failed: {e}")
        # Attempt to clean up even if training failed
        if os.path.exists(TEMP_DIR):
             shutil.rmtree(TEMP_DIR)
        return None, None, None

# --- STREAMLIT UI ---

st.title("🌿 Groundnut Tikka Leaf Disease Detection System")
st.markdown("### AI-Powered Disease Identification and Severity Scoring")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Predict Disease", "Train Model", "About"])

# Main pages
if page == "Predict Disease":
    st.header("📸 Upload Groundnut Leaf Images")
    
    model, class_indices = load_model()
    
    if model is None:
        st.warning("⚠️ No trained model found. Please train a model first in the 'Train Model' section.")
    else:
        st.success("✅ Model loaded successfully!")
        
        uploaded_files = st.file_uploader(
            "Choose groundnut leaf images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.markdown("---")
            results = []
            
            # Create columns for layout
            cols = st.columns(2)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    # Predict
                    pred_class, confidence, severity_name, all_probs = predict_image(
                        model, image, class_indices
                    )
                    
                    # Display results
                    col = cols[idx % 2]
                    
                    with col:
                        st.image(image, caption=uploaded_file.name, use_container_width=True)
                        
                        # Display prediction
                        severity_color = SEVERITY_COLORS.get(pred_class, "gray")
                        st.markdown(f"**Status:** :{severity_color}[{severity_name}]")
                        st.markdown(f"**Confidence:** {confidence:.2f}%")
                        
                        # Progress bar for confidence
                        st.progress(confidence / 100)
                        
                        # Recommendations
                        if pred_class == 0:
                            st.info("✅ Leaf is healthy. Continue regular monitoring.")
                        elif pred_class == 1:
                            st.warning("⚠️ Mild disease detected. Monitor closely and consider preventive measures.")
                        elif pred_class == 2:
                            st.warning("⚠️ Moderate disease. Apply fungicide treatment recommended for Tikka disease.")
                        else: # pred_class == 3
                            st.error("🚨 Severe disease. Immediate fungicide treatment required. Consult agricultural expert.")
                        
                        # Show probability distribution
                        with st.expander("View detailed probability scores"):
                            
                            prob_data = []
                            for i, prob in enumerate(all_probs):
                                display_name = SEVERITY_LEVELS.get(i, f"Class Index {i}")
                                
                                prob_data.append({
                                    'Severity Level': display_name,
                                    'Probability (%)': prob * 100
                                })
                                
                            prob_df = pd.DataFrame(prob_data)
                            st.dataframe(prob_df, use_container_width=True)
                        
                        st.markdown("---")
                    
                    # Store results
                    results.append({
                        'Image': uploaded_file.name,
                        'Status': severity_name,
                        'Confidence (%)': f"{confidence:.2f}",
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            # Summary and download
            if results:
                st.markdown("---")
                st.subheader("📊 Batch Summary")
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name=f"tikka_disease_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv'
                )

elif page == "Train Model":
    st.header("🎓 Train Disease Detection Model")
    
    st.markdown("""
    ### Training Dataset Instructions:
    1. **Format:** Create a **ZIP file** containing your dataset.
    2. **Structure:** Ensure the ZIP file extracts to a folder containing subfolders for each class:
    ```
    my_dataset.zip
    └── training_data/
        ├── healthy/
        ├── mild/
        ├── moderate/
        └── severe/
    ```
    """)
    
    # Option 1: File Uploader (Ideal for deployment)
    uploaded_zip_file = st.file_uploader(
        "Upload your dataset as a ZIP file (Recommended)", 
        type=['zip']
    )
    
    # Option 2: Local Path (Fallback for local development)
    st.markdown("---")
    st.markdown("#### **Local Training Path (For local development only)**")
    # Using the path specified in previous context, but making it optional
    local_training_path = st.text_input(
        "Enter path to your data folder (if not uploading a ZIP)", 
        value=r"F:\ML_images\Trainingdata" # Retain the specific path for convenience
    )
    
    epochs = st.slider("Number of Epochs", 5, 50, 10)
    
    if st.button("🚀 Start Training"):
        model, history, class_indices = None, None, None
        training_source = None
        
        if uploaded_zip_file:
            # Training via ZIP upload
            training_source = "ZIP Upload"
            with st.spinner("Processing ZIP file and starting training..."):
                model, history, class_indices = unzip_and_train(uploaded_zip_file, epochs)
        
        elif local_training_path and os.path.exists(local_training_path):
            # Training via Local Path
            training_source = "Local Path"
            with st.spinner("Starting training from local path..."):
                try:
                    model, history, class_indices = train_model(local_training_path, epochs)
                except (FileNotFoundError, ValueError) as e:
                    st.error(f"❌ Training setup failed: {e}")
                except Exception as e:
                    st.error(f"❌ Training execution failed: {str(e)}")
        
        else:
            st.error("❌ Please either **upload a ZIP file** or ensure the **Local Training Path** is correct and accessible.")
            
        # Display results if training was successful
        if model is not None:
            save_model(model, class_indices)
            
            st.success(f"✅ Model trained and saved successfully using {training_source}!")
            
            st.subheader("Model Performance Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Final Training Accuracy", f"{history.history['accuracy'][-1]*100:.2f}%")
            
            with col2:
                st.metric("Final Validation Accuracy", f"{history.history['val_accuracy'][-1]*100:.2f}%")
            
            st.subheader("Training History")
            history_df = pd.DataFrame({
                'Epoch': range(1, epochs + 1),
                'Training Accuracy': history.history['accuracy'],
                'Validation Accuracy': history.history['val_accuracy'],
                'Training Loss': history.history['loss'],
                'Validation Loss': history.history['val_loss']
            })
            
            st.line_chart(history_df.set_index('Epoch')[['Training Accuracy', 'Validation Accuracy']])
            st.line_chart(history_df.set_index('Epoch')[['Training Loss', 'Validation Loss']])

else:  # About page
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### Groundnut Tikka Disease Detection System
    
    This application uses deep learning to identify and assess the severity of **Tikka disease** (also known as leaf spot disease) in groundnut (peanut) plants.
    
    #### Features:
    - **AI-Powered Detection**: Uses **MobileNetV2** transfer learning for accurate classification
    - **Severity Scoring**: Classifies disease into 4 levels (Healthy, Mild, Moderate, Severe)
    - **Batch Processing**: Analyze multiple leaf images simultaneously
    - **Confidence Scores**: Provides probability scores for each prediction
    - **Actionable Recommendations**: Suggests treatment based on disease severity
    - **Custom Training**: Train on your own dataset for improved accuracy
    
    #### About Tikka Disease:
    Tikka disease is caused by the fungus *Cercosporidium personatum*. It appears as:
    - Small circular spots on leaves
    - Brown to black lesions
    - Yellow halos around spots
    - Progressive leaf damage if untreated
    
    
    #### How to Use:
    1. **Predict Disease**: Upload leaf images for instant analysis
    2. **Train Model**: **Upload your dataset as a ZIP file** or use a local path to customize the model.
    3. **Download Results**: Export predictions for record keeping
    
    #### Technical Details:
    - **Model Architecture**: MobileNetV2 with custom classification head
    - **Input Size**: 224x224 pixels
    - **Data Augmentation**: Rotation, flip, zoom, shift for robust training
    - **Framework**: TensorFlow/Keras
    
    ---
    
    **Note**: This is a diagnostic tool. For critical agricultural decisions, 
    consult with agricultural experts and plant pathologists.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Tips for Best Results:**
- Use clear, well-lit images
- Focus on affected leaf areas
- Capture multiple angles
- Ensure leaves are in focus

""")
