import firebase_admin
from firebase_admin import credentials
import pyrebase
import os
import json

# Path to service account key (for Firebase Admin, optional for auth)
FIREBASE_CONFIG_PATH = os.path.join("src", "firebase", "firebase_config.json")
cred = credentials.Certificate(FIREBASE_CONFIG_PATH)

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Web API Config (from Firebase Console → Project Settings → Web App SDK setup)
firebase_web_config = {
    "apiKey": "AIzaSyDJS3xr5elOrVeTI0VmcBMA9adKt-Kxdyw",
    "authDomain": "automated-ml-platform.firebaseapp.com",
    "databaseURL": "https://automated-ml-platform.firebaseio.com", 
    "projectId": "automated-ml-platform",
    "storageBucket": "automated-ml-platform.appspot.com",
    "messagingSenderId": "553548923511",
    "appId": "1:553548923511:web:e82b0ebf8fb8118f1bfc72"
}

# Initialize Pyrebase with correct config
firebase = pyrebase.initialize_app(firebase_web_config)
auth = firebase.auth()

def signup_user(email, password):
    try:
        user = auth.create_user_with_email_and_password(email, password)
        return user
    except Exception as e:
        return str(e)

def login_user(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        return user
    except Exception as e:
        return str(e)
