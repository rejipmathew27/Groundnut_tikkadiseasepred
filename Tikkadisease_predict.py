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

# Set page config
st.set_page_config(
    page_title="Groundnut Tikka Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
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

# Model building function
def create_model(num_classes=4):
    """Create a transfer learning model using MobileNetV2"""
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
    
    train_generator = train_datagen.flow_from_directory(
        training_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
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
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    # Get class name
    class_names = {v: k for k, v in class_indices.items()}
    severity_name = class_names.get(predicted_class, f"Class {predicted_class}")
    
    return predicted_class, confidence, severity_name, predictions[0]

# Save/Load model functions
def save_model(model, class_indices):
    """Save model and class indices"""
    model.save('tikka_disease_model.h5')
    np.save('class_indices.npy', class_indices)
    
def load_model():
    """Load saved model and class indices"""
    if os.path.exists('tikka_disease_model.h5') and os.path.exists('class_indices.npy'):
        model = keras.models.load_model('tikka_disease_model.h5')
        class_indices = np.load('class_indices.npy', allow_pickle=True).item()
        return model, class_indices
    return None, None

# Streamlit UI
st.title("🌿 Groundnut Tikka Leaf Disease Detection System")
st.markdown("### AI-Powered Disease Identification and Severity Scoring")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Predict Disease", "Train Model", "About"])

# Main pages
if page == "Predict Disease":
    st.header("📸 Upload Groundnut Leaf Images")
    
    # Load model
    model, class_indices = load_model()
    
    if model is None:
        st.warning("⚠️ No trained model found. Please train a model first in the 'Train Model' section.")
    else:
        st.success("✅ Model loaded successfully!")
        
        # File uploader
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
                        else:
                            st.error("🚨 Severe disease. Immediate fungicide treatment required. Consult agricultural expert.")
                        
                        # Show probability distribution
                        with st.expander("View detailed probability scores"):
                            # Map numerical indices to severity names
                            class_map = {v: SEVERITY_LEVELS.get(int(k), f"Class {k}") for k, v in class_indices.items()}
                            
                            prob_data = []
                            # Ensure all classes in class_indices are included and probabilities are aligned
                            sorted_indices = sorted(class_indices.values())
                            sorted_class_names = [class_map.get(i, f"Class {i}") for i in sorted_indices]
                            
                            for i in sorted_indices:
                                # Find the original class name (key) corresponding to the index (value)
                                original_class_key = next((k for k, v in class_indices.items() if v == i), f"Class {i}")
                                # Use SEVERITY_LEVELS for display if available, fallback to original key
                                display_name = SEVERITY_LEVELS.get(i, original_class_key)
                                
                                prob_data.append({
                                    'Severity Level': display_name,
                                    'Probability (%)': all_probs[i] * 100
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
    ### Training Instructions:
    1. Prepare your dataset with the following folder structure:
    ```
    training_data/
    ├── healthy/
    ├── mild/
    ├── moderate/
    └── severe/
    ```
    2. Enter the path to your **training\_data** folder
    3. Configure training parameters
    4. Click 'Start Training'
    """)
    
    # MODIFICATION: Set the default value to the desired path
    training_path = st.text_input("Training Data Path", **r"F:\ML_images\Trainingdata"**) 
    epochs = st.slider("Number of Epochs", 5, 50, 10)
    
    if st.button("🚀 Start Training"):
        if not os.path.exists(training_path):
            st.error(f"❌ Path not found: {training_path}")
        else:
            with st.spinner("Training in progress... This may take several minutes."):
                try:
                    model, history, class_indices = train_model(training_path, epochs)
                    
                    # Save model
                    save_model(model, class_indices)
                    
                    st.success("✅ Model trained and saved successfully!")
                    
                    # Display training results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Final Training Accuracy", f"{history.history['accuracy'][-1]*100:.2f}%")
                    
                    with col2:
                        st.metric("Final Validation Accuracy", f"{history.history['val_accuracy'][-1]*100:.2f}%")
                    
                    # Plot training history
                    st.subheader("Training History")
                    history_df = pd.DataFrame({
                        'Epoch': range(1, epochs + 1),
                        'Training Accuracy': history.history['accuracy'],
                        'Validation Accuracy': history.history['val_accuracy']
                    })
                    st.line_chart(history_df.set_index('Epoch'))
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")

else:  # About page
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### Groundnut Tikka Disease Detection System
    
    This application uses deep learning to identify and assess the severity of Tikka disease 
    (also known as leaf spot disease) in groundnut (peanut) plants.
    
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
    2. **Train Model**: Customize the model with your own dataset
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
