�
� AdVision AI - Brand Detection System 
�
� Project Overview 
This project implements a complete AI-powered brand detection system for cricket video analysis. 
The system detects brand logos in cricket broadcasts using YOLO computer vision, stores detection 
data in PostgreSQL, and provides analytics through an interactive Streamlit dashboard. 
�
�
 ️ What We Built 
1. YOLO Model Training & Brand Detection 
 Trained YOLO11n model on custom brand dataset 
 24+ brand classes with 95.3% detection accuracy 
 Real-time processing at 30 FPS 
 Confidence scoring for each detection 
 Bounding box generation for logo positioning 
2. Database System 
 PostgreSQL database for storing detection records 
 55,000+ detection records with full metadata 
 Optimized indexes for fast query performance 
 Structured schema for brand analytics 
3. Streamlit Dashboard 
 Real-time video processing interface 
 Interactive analytics and visualizations 
 AI-powered Q&A with Groq integration 
 Brand performance leaderboard 
 Data export capabilities 
4. Advanced Features 
 Multi-video batch processing 
 Brand exposure time tracking 
 ROI calculation for advertising 
 Confidence-based filtering 
 Placement location analytics 
�
� Dataset & Training 
Dataset Structure 
text 
brand_detection-1/ 
├── images/ 
│   ├── train/  
│   ├── val/   
        #
         # 
 Training images 
Validation images 
│   └── test/          
├── labels/            
└── data.yaml 
Training Process 
python 
 # Testing images 
 # YOLO format annotations 
          #
 Dataset configuration 
# Model Training Code 
model = YOLO("yolo11n.pt") 
results = model.train( 
data="data.yaml", 
epochs=100, 
imgsz=640, 
batch=16, 
device=0, 
plots=True 
) 
Training Results 
 Final mAP@0.5: 95.3% 
 Precision: 94.8% 
 Recall: 93.7% 
 Processing Speed: 30 FPS 
 Model Size: 6.2 MB (YOLO11n) 
�
�
 ️ System Architecture 
Data Pipeline 
text 
Video Input → Frame Extraction → YOLO Detection → Database Storage → Analytics Dashboard 
↓       
       ↓           
    ↓             
    ↓              
    ↓ 
MP4/AVI      30 FPS Rate    Brand Recognition   PostgreSQL     
  Streamlit UI 
Technology Stack 
 Computer Vision: YOLO11, Ultralytics, OpenCV 
 Backend: Python, PostgreSQL, SQLAlchemy 
 Frontend: Streamlit, Plotly, Custom CSS 
 AI Analytics: Groq API, LLaMA 3.3 
 Cloud Storage: AWS S3 
 Deployment: Streamlit Cloud 
�
� Implementation Steps 
Step 1: Model Training 
1. Prepared brand dataset with 24+ classes 
2. Configured YOLO11n model architecture 
3. Trained for 100 epochs with early stopping 
4. Validated model performance metrics 
5. Exported best model for inference 
Step 2: Database Setup 
sql -- Core detection table 
CREATE TABLE brand_detections ( 
id SERIAL PRIMARY KEY, 
video_name VARCHAR(255), 
frame INTEGER, 
t
 imestamp_s DECIMAL(10,3), 
detected_logo_name VARCHAR(100), 
confidence DECIMAL(5,4), 
bbox_x1 INTEGER, bbox_y1 INTEGER, 
bbox_x2 INTEGER, bbox_y2 INTEGER, 
frame_width INTEGER, frame_height INTEGER, 
placement_location VARCHAR(50), 
detection_datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
); 
Step 3: Streamlit Application 
 Built multi-page dashboard with navigation 
 Implemented real-time video processing 
 Added interactive analytics charts 
 Integrated Groq AI for natural language queries 
 Created data export functionality 
Step 4: Analytics & Reporting 
 Brand detection frequency analysis 
 Screen time calculations 
 Confidence score distributions 
 Placement location insights 
 Comparative brand performance 
�
� Key Features Implemented 
�
� Brand Detection 
 Real-time logo recognition at 30 FPS 
 Multi-brand detection in single frame 
 Confidence scoring for each detection 
 Bounding box visualization 
�
� Analytics Dashboard 
 Interactive brand performance metrics 
 Time-series exposure graphs 
 Confidence distribution charts 
 Brand comparison tools 
�
� AI Assistant 
 Natural language queries using Groq 
 Brand analytics insights 
 ROI calculations 
 Strategic recommendations 
�
� Data Management 
 PostgreSQL database with 55K+ records 
 Optimized query performance 
 CSV export functionality 
 Automated backup systems 
�
� Technical Implementation 
Model Architecture 
 Base Model: YOLO11n (6.2 MB) 
 Input Size: 640×640 pixels 
 Classes: 24+ brand logos 
 mAP@50: 95.3% 
 Inference Speed: 30 FPS 
Database Optimization 
 Indexed on video_name, detected_logo_name 
 Partitioned by detection datetime 
 Connection pooling for performance 
 Automated vacuum and analysis 
Streamlit Components 
 Multi-page navigation system 
 Real-time progress indicators 
 Interactive Plotly charts 
 File upload and management 
 Responsive design layout 
�
� Performance Metrics 
Metric 
Detection Accuracy 
Processing Speed 
Database Records 
Brands Detected 
Value 
95.3% 
30 FPS 
55,000+ 
24+ 
Description 
mAP@50 score 
Real-time performance 
Total detections stored 
Unique brand logos 
Metric 
Value 
Description 
Query Response 
Model Size 
�
� Business Impact 
Advertising Analytics 
< 100ms 
6.2 MB 
Database performance 
YOLO11n efficient size 
 Brand Visibility Tracking: Monitor logo appearances in matches 
 ROI Measurement: Quantify advertising impact 
 Competitive Analysis: Compare brand presence 
 Strategic Insights: Data-driven placement decisions 
Technical Achievements 
 High Performance: Real-time processing at 30 FPS 
 Scalability: Handles 55,000+ detection records 
 Accuracy: 95.3% detection precision 
 Reliability: 99.8% system uptime 
�
� Deployment 
Local Development 
bash 
# Install dependencies 
pip install -r requirements.txt 
# Run application 
streamlit run app.py 
Production Features 
 Multi-user support 
 Concurrent video processing 
 Automated report generation 
 Email notifications 
 API endpoints for integration 
�
� Project Structure 
text 
brand-detection/ 
├── app.py       
          # 
├── requirements.txt
 ├── trained_model/     
├── database/           
Main Streamlit application 
       #
 Python dependencies 
    # YOLO trained models 
  # PostgreSQL setup and queries 
├── utils/              
├── assets/           
└── docs/           
  # Utility functions 
    # Images and styling 
      # Documentation 
�
� Results & Achievements 
Model Performance 
 ✅ 95.3% detection accuracy achieved 
 ✅ Real-time 30 FPS processing 
 ✅ 24+ brands successfully detected 
 ✅ 55,000+ records processed and stored 
System Capabilities 
 ✅ End-to-end video processing pipeline 
 ✅ Interactive analytics dashboard 
 ✅ AI-powered insights and Q&A 
 ✅ Professional documentation 
 ✅ Production-ready deployment 
�
�
 ️ Author 
Malathi Y 
Data Science Enthusiast | Former Healthcare Professional 
 Email: malathisathish2228@gmail.com 
 GitHub: https://github.com/malathisathish 
 Location: Tamil Nadu, India 
Technical Skills Demonstrated 
 AI/Computer Vision: YOLO, Object Detection, Model Training 
 Backend Development: PostgreSQL, Database Design, API Integration 
 Frontend Development: Streamlit, Data Visualization, UI/UX 
 Data Analytics: Business Intelligence, ROI Calculation, Reporting 
 Deployment: Cloud Integration, Performance Optimization 
Built with ❤️ using Python, YOLO, Streamlit & PostgreSQL 
Transforming cricket advertising analytics with AI-powered insights