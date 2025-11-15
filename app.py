import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="White Blood Cell Classification Using Deep Learning", page_icon="🧬")
st.title("🧬 White Blood Cell Classification Using Deep Learning")

# Short and Crisp Introduction
st.write("""
White blood cells (WBCs) are crucial components of the immune system that protect the body from infections and diseases.  
They are classified into four main types:

- **Neutrophils** — the first responders that fight bacterial and fungal infections.  
- **Lymphocytes** — produce antibodies and destroy virus-infected or abnormal cells.  
- **Monocytes** — act as clean-up cells, removing dead tissue and attacking pathogens.  
- **Eosinophils** — help control allergic reactions and combat parasitic infections.  

Classification of these cells from microscopic images aids in the **early detection of infections, allergies, and blood disorders**,  
enabling faster and more reliable medical diagnosis.
""")

st.write("Upload a microscopic image of a white blood cell to identify its type and learn its biological role below.")

model = tf.keras.models.load_model("model_name.h5")
CLASS_NAMES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
IMG_SIZE = (224, 224)

CELL_ROLES = {
    "EOSINOPHIL": "Eosinophils help control allergic reactions and defend the body against parasitic infections.",
    "LYMPHOCYTE": "Lymphocytes coordinate the immune response — B cells produce antibodies and T cells destroy virus-infected cells.",
    "MONOCYTE": "Monocytes are the body’s clean-up crew — they remove dead cells and become macrophages that digest pathogens.",
    "NEUTROPHIL": "Neutrophils act as the first line of defense by quickly attacking and destroying bacteria and fungi."
}

uploaded_file = st.file_uploader("Upload a WBC image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)  # Auto-fix rotation if needed

    # Layout: show uploaded image on left, predictions on right
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📸 Uploaded Image")
        st.image(img, width=300, use_container_width=True)

    # Preprocess image for model
    arr = np.expand_dims(np.array(img.resize(IMG_SIZE)) / 255.0, axis=0)
    preds = model.predict(arr)[0]
    predicted_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = preds[predicted_index] * 100

    with col2:
        st.markdown("### Prediction Result")

        st.markdown(
                    f"<h4><b>Predicted Cell Type:</b> "
                    f"<span style='font-size:26px; color:#00FFAA;'>{predicted_class}</span></h4>",
        unsafe_allow_html=True
        )

        st.markdown(
        f"<h4><b>Model Confidence:</b> {confidence:.2f}%</h4>",
        unsafe_allow_html=True
        )

    st.info(CELL_ROLES[predicted_class])
<<<<<<< HEAD
    
    # Add note for clarity
    st.markdown("---")
    
=======
    # 🧾 Add note for clarity
    st.markdown("---")
>>>>>>> 9ba661f6d08ebdca33dcc0bdc82b881ee2096239
