import streamlit as st
import streamlit_analytics
import os
import json
import tempfile
from dotenv import load_dotenv

# Load environment variables for local development
load_dotenv()

def get_firebase_key_path():
    """
    Retrieves the Firebase key path from Streamlit secrets or environment variables.
    For deployed apps, writes the JSON key to a temporary file and returns the path.
    For local development, returns the path from FIREBASE_KEY_PATH.
    """
    # Check if FIREBASE_KEY_JSON exists in Streamlit secrets
    if 'FIREBASE_KEY_JSON' in st.secrets:
        firebase_key_json = st.secrets['FIREBASE_KEY_JSON']
        
        # Create a temporary file to store the JSON key
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_key_file:
            temp_key_file.write(firebase_key_json)
            temp_key_path = temp_key_file.name
        
        return temp_key_path
    
    # For local development, use FIREBASE_KEY_PATH from environment variables
    firebase_key_path = os.getenv("FIREBASE_KEY_PATH")
    if firebase_key_path and os.path.exists(firebase_key_path):
        return firebase_key_path
    
    # If neither method works, raise an error
    raise ValueError("Firebase key path not found in secrets or local environment")

def main():
    # Retrieve Firebase credentials
    try:
        firebase_key_path = get_firebase_key_path()
        firebase_collection = st.secrets.get('FIREBASE_COLLECTION', 'counts')
    except ValueError as e:
        st.error(str(e))
        st.stop()
    
    # Initialize Firebase Analytics Tracking
    try:
        streamlit_analytics.track(
            firestore_key_file=firebase_key_path,
            firestore_collection_name=firebase_collection
        )
    except Exception as e:
        st.error(f"Error initializing analytics tracking: {e}")
        st.stop()
    
    st.title("Step 1 Buddy")

    # Your existing app code goes here
    # ...

if __name__ == "__main__":
    main()
