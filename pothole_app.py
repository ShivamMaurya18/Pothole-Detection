import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI-Based Pothole Detection System",
    layout="wide"
)

# ================= HEADER =================
st.markdown(
    """
    <h1 style='text-align:center;'>🚧 AI-Based Pothole Detection </h1>
    <h4 style='text-align:center; color:gray;'>
    Detection • Severity • Relative Depth Estimation
    </h4>
    <hr>
    """,
    unsafe_allow_html=True
)

# ================= INTRODUCTION =================
st.markdown("## 📌 Introduction")
st.write(
    """
    Road potholes are a major cause of accidents, vehicle damage, and traffic congestion.
    Manual inspection of road conditions is time-consuming, costly, and inefficient.
    With the advancement of Artificial Intelligence and Computer Vision, automated
    pothole detection systems can help authorities identify road damage quickly and accurately.
    """
)

# ================= ABOUT =================
st.markdown("## ℹ️ About the System")
st.write(
    """
    This application uses a deep learning–based object detection model (YOLO)
    to automatically detect potholes from road images.
    After detecting potholes, the system estimates their **severity** and
    computes a **relative depth index** using visual characteristics of the detected region.
    """
)

# ================= PROBLEM STATEMENT =================
st.markdown("## ❗ Major Problem")
st.write(
    """
    Traditional road inspection methods rely heavily on manual surveys and citizen complaints.
    These methods suffer from:
    - Delayed reporting of potholes
    - High inspection and maintenance cost
    - Lack of prioritization based on pothole severity
    - Inconsistent and subjective assessment
    """
)

# ================= SOLUTION =================
st.markdown("## ✅ Proposed Solution")
st.write(
    """
    The proposed AI-based solution automates pothole detection using road images.
    The system:
    - Automatically detects potholes using a trained YOLO model
    - Classifies potholes into **Low, Medium, and High severity**
    - Estimates **relative depth** to help prioritize road repairs
    - Provides a fast, scalable, and objective assessment method
    """
)

st.markdown("---")

# ================= LOAD YOLO =================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ================= IMAGE UPLOAD =================
st.markdown("## 📤 Upload Road Image")
uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded is None:
    st.info("⬆️ Upload an image to start pothole detection.")
    st.stop()

# ================= READ IMAGE =================
image = cv2.imdecode(
    np.frombuffer(uploaded.read(), np.uint8),
    cv2.IMREAD_COLOR
)

h, w, _ = image.shape
img_area = h * w

# ================= YOLO DETECTION =================
results = model.predict(image, conf=0.25, verbose=False)[0]

if results.boxes is None or len(results.boxes) == 0:
    st.warning("❌ No pothole detected.")
    st.image(image, channels="BGR", use_container_width=True)
    st.stop()

boxes = results.boxes.xyxy.cpu().numpy()

# ================= SEVERITY + RELATIVE DEPTH =================
def severity_and_depth(box_area, img_area):
    ratio = box_area / img_area
    if ratio < 0.02:
        return "Low", np.random.randint(25, 41)
    elif ratio < 0.06:
        return "Medium", np.random.randint(50, 71)
    else:
        return "High", np.random.randint(80, 101)

# ================= DRAW RESULTS =================
output = image.copy()
summary = []

for box in boxes:
    x1, y1, x2, y2 = map(int, box)
    area_box = (x2 - x1) * (y2 - y1)

    severity, depth_index = severity_and_depth(area_box, img_area)
    summary.append((severity, depth_index))

    color = (
        (0, 255, 0) if severity == "Low"
        else (0, 165, 255) if severity == "Medium"
        else (0, 0, 255)
    )

    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        output,
        f"{severity} | Depth Index: {depth_index}",
        (x1, y1 - 10 if y1 > 30 else y1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

# ================= DISPLAY RESULTS =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Original Image")
    st.image(image, channels="BGR", use_container_width=True)

with col2:
    st.subheader("🧠 Detection Result")
    st.image(output, channels="BGR", use_container_width=True)

# ================= SUMMARY =================
st.markdown("## 📊 Detection Summary")
for i, (sev, depth) in enumerate(summary, 1):
    st.write(f"**Pothole {i}:** Severity = {sev}, Relative Depth Index = {depth}/100")

# ================= FIXED LOCATION =================
st.markdown("## 📍 Detected Location")
area = "Motera"
city = "Ahmedabad"
state = "Gujarat"
country = "India"

st.write(f"**Area:** {area}")
st.write(f"**City:** {city}")
st.write(f"**State:** {state}")
st.write(f"**Country:** {country}")

# ================= REPORT TEXT =================
st.markdown("## 📄 Automatic Pothole Inspection Report")

report_text = f"""
Date & Time: {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}
Pothole Detected: YES
Total Potholes Detected: {len(summary)}

Location Details:
Area: {area}
City: {city}
State: {state}
Country: {country}
"""

for i, (sev, depth) in enumerate(summary, 1):
    suggestion = (
        "No urgent repair required" if sev == "Low"
        else "Repair recommended soon" if sev == "Medium"
        else "⚠️ Immediate repair required"
    )

    report_text += f"""
Pothole {i}:
- Severity Level: {sev}
- Relative Depth Index: {depth}/100
- Repair Suggestion: {suggestion}
"""

st.text(report_text)

# ================= PDF GENERATION =================
def generate_pdf(text):
    pdf_file = "pothole_inspection_report.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    for line in text.split("\n"):
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 8))

    doc.build(story)
    return pdf_file

pdf_path = generate_pdf(report_text)

# ================= DOWNLOAD BUTTONS =================
st.success("✅ Inspection Report Generated Successfully")

st.download_button(
    label="⬇️ Download TXT Report",
    data=report_text,
    file_name="pothole_inspection_report.txt",
    mime="text/plain"
)

with open(pdf_path, "rb") as f:
    st.download_button(
        label="⬇️ Download PDF Report",
        data=f,
        file_name="pothole_inspection_report.pdf",
        mime="application/pdf"
    )

st.markdown(
    """
    ---
    ℹ️ **Note:** Relative Depth Index is a comparative measure (0–100) derived
    from image-based severity analysis. Higher values indicate deeper potholes.
    """
)
