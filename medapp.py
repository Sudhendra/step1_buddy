import streamlit as st
import streamlit_analytics
import os
import json
import tempfile
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables for local development
load_dotenv()

def get_firebase_key():
    """
    Retrieves the Firebase key from Streamlit secrets or environment variables.
    For deployed apps, returns the JSON key directly or writes it to a temporary file.
    For local development, returns the path from FIREBASE_KEY_PATH.
    """
    logging.info("Attempting to retrieve Firebase key...")
    logging.info(f"Available secrets: {', '.join(st.secrets.keys())}")

    if 'FIREBASE_KEY_JSON' in st.secrets or 'firebase_key_json' in st.secrets:
        firebase_key_json = st.secrets.get('FIREBASE_KEY_JSON') or st.secrets.get('firebase_key_json')
        logging.info(f"First 20 characters of FIREBASE_KEY_JSON: {firebase_key_json[:20]}...")
        
        try:
            json.loads(firebase_key_json)
            logging.info("Successfully parsed FIREBASE_KEY_JSON as JSON")
            return firebase_key_json
        except json.JSONDecodeError:
            logging.info("FIREBASE_KEY_JSON is not valid JSON")
    
    # For local development, use FIREBASE_KEY_PATH from environment variables
    firebase_key_path = os.getenv("FIREBASE_KEY_PATH")
    if firebase_key_path and os.path.exists(firebase_key_path):
        logging.info(f"Using local Firebase key path: {firebase_key_path}")
        return firebase_key_path
    
    # If neither method works, raise an error
    logging.error("Firebase key not found in secrets or local environment")
    raise ValueError("Firebase key not found in secrets or local environment")

def main():
    # Retrieve Firebase credentials
    try:
        firebase_key = get_firebase_key()
        firebase_collection = st.secrets.get('FIREBASE_COLLECTION', 'counts')
        logging.info(f"Firebase collection: {firebase_collection}")
    except ValueError as e:
        st.error(str(e))
        logging.error(f"Error retrieving Firebase key: {str(e)}")
        st.stop()
    
    # Initialize Firebase Analytics Tracking
    try:
        if isinstance(firebase_key, str) and firebase_key.startswith('{'):
            # If firebase_key is a JSON string, write it to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
                temp_file.write(firebase_key)
                firebase_key_path = temp_file.name
        else:
            firebase_key_path = firebase_key

        logging.info(f"Initializing tracking with key path: {firebase_key_path}")
        streamlit_analytics.track(
            firestore_key_file=firebase_key_path,
            firestore_collection_name=firebase_collection
        )
        logging.info("Successfully initialized analytics tracking")
    except Exception as e:
        st.error(f"Error initializing analytics tracking: {e}")
        logging.error(f"Error initializing analytics tracking: {str(e)}")
        st.stop()
    
    st.title("Step 1 Buddy")

    # Your existing app code goes here
    # ...

if __name__ == "__main__":
    main()
