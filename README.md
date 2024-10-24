# Step Zero

Step Zero is an AI-powered application designed to assist medical students in their preparation for the USMLE Step 1 exam. The app uses natural language processing and machine learning techniques to provide relevant information from video lectures based on user queries.

## Visit Us
URL: https://stepzero.streamlit.app/

![streamlit-medapp-2024-10-24-19-10-28 (1)](https://github.com/user-attachments/assets/5b610412-9ad4-480f-8659-624b05c54b21)


## Features

- Topic-based question answering
- Retrieval of relevant passages from video lectures
- RAG-generated answers using OpenAI's GPT-4
- Frame extraction from videos at specific timestamps
- Interactive Mindmap Generation
- Critical analysis report generation for step 1 success
- Feedback action

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Sudhendra/step1_buddy.git
   cd step1-buddy
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key as an environment variable or input it in the app's sidebar.

## Local Usage

Run the Streamlit app:

```
streamlit run medapp.py
```

Navigate to the provided local URL in your web browser to use the application.

## Data

The application uses video data stored in JSON format. Each topic has its own JSON file in the `data` directory. The app also generates and caches embeddings for faster retrieval.

## Disclosures

Please refer to the "Disclosures" tab in the application for important information about the source of the content, licensing, and future updates.

## License

This project is protected by license. Please contact the repository owner for permission before using the code for personal or commercial projects.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
