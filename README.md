🎯 AdVision AI — Intelligent Brand Detection System

📋 Project Overview

AdVision AI is an end-to-end AI-powered brand detection system built for analyzing cricket broadcasts.
It uses YOLO-based computer vision to detect brand logos in video frames, stores detections in a PostgreSQL database,
and delivers actionable insights through a sleek Streamlit dashboard.

✨ Features

🤖 AI-Powered Detection

Real-time 30 FPS processing, 95.3% mAP@50 accuracy, multi-brand recognition (24+ classes),
and bounding box visualization with precision scoring.

📊 Advanced Analytics

Brand performance dashboards, ROI measurement, exposure tracking, and placement intelligence
for detailed marketing insights.

💾 Data Management

PostgreSQL backend storing 55,000+ detections, optimized queries (<100ms response time),
CSV export, and AWS S3 integration for cloud storage.

🎯 User Experience

Streamlit dashboard with multi-page navigation, Groq-powered AI assistant, and live detection overlay.

🚀 Quick Start

1. Clone the repository and install dependencies

2. Configure PostgreSQL and environment variables

3. Run `streamlit run app.py` to start the application

🏗️ System Architecture

Video Input → Frame Extraction → YOLO Detection → Database Storage → Analytics Dashboard

📁 Project Structure

advision-ai/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
├── model                  # YOLO trained models
├── .env                   # Paswords and credentials stored 
├── readme.md              # Readme file 
└── project_documentation  # Documentation files

🎯 Use Cases

🏏 Sports Broadcasting — Real-time brand tracking & performance analytics

📊 Marketing Analytics — ROI measurement for sponsorships

🔍 Media Monitoring — Compliance and brand exposure benchmarking

💼 Business Impact


- 92% reduction in manual monitoring costs

- 10x faster data processing pipeline

- Real-time data-driven insights

🔧 Configuration

MODEL_PATH: YOLO model path

PG_*: PostgreSQL credentials

AWS_*: S3 credentials

GROQ_API_KEY: Groq AI integration key

Supported Video Formats: MP4, AVI, MOV, MKV (Up to 4K @ 30 FPS)

🚀 Deployment Options

- Local: `streamlit run app.py`

- Cloud: Streamlit Cloud, AWS EC2, Heroku, Google Cloud

- Production: Auto-scaling, connection pooling, S3-based storage

🤝 Contributing

Fork the repository → Create a feature branch → Commit → Submit Pull Request

📄 License
This project is licensed under the MIT License.

🆘 Support

📧 Email: malathisathish2228@gmail.com

💬 GitHub: https://github.com/malathisathish

📚 Docs: /docs folder

👩‍💻 Author

Malathi Y — Data Science Enthusiast | Former Healthcare Professional

Email: malathisathish2228@gmail.com

GitHub: malathisathish

Location: Tamil Nadu, India

Built with ❤️ using Python, YOLO, Streamlit & PostgreSQL

Transforming cricket advertising analytics through AI-powered insights.

