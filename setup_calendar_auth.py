#!/usr/bin/env python3
"""
Google Calendar Authentication Setup Script

This script helps you set up Google Calendar API authentication properly.
"""

import os
import json
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

def setup_calendar_auth():
    """Set up Google Calendar authentication"""
    
    print("🔧 Google Calendar Authentication Setup")
    print("=" * 50)
    
    # Check if credentials file exists
    creds_path = 'client_secret_287440476338-njd85ng1uu56ontjt46gllm6qgtitl5d.apps.googleusercontent.com.json'
    token_path = 'calendar_token.json'
    
    if not os.path.exists(creds_path):
        print("❌ Credentials file not found!")
        print(f"Looking for: {creds_path}")
        print("\n📋 To get this file:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project or select existing one")
        print("3. Enable Google Calendar API")
        print("4. Go to 'APIs & Services' → 'Credentials'")
        print("5. Click 'Create Credentials' → 'OAuth 2.0 Client IDs'")
        print("6. Choose 'Desktop application'")
        print("7. Add these redirect URIs:")
        print("   - urn:ietf:wg:oauth:2.0:oob")
        print("   - http://localhost")
        print("8. Download the JSON file and place it in the same directory as this script")
        return False
    
    try:
        # Read credentials file
        with open(creds_path, 'r') as f:
            creds_data = json.load(f)
        
        # Get redirect URI from credentials
        redirect_uri = creds_data.get('installed', {}).get('redirect_uris', ['urn:ietf:wg:oauth:2.0:oob'])[0]
        
        print(f"✅ Found credentials file")
        print(f"📋 Using redirect URI: {redirect_uri}")
        
        # Check if token already exists
        if os.path.exists(token_path):
            print("✅ Found existing token file")
            creds = Credentials.from_authorized_user_file(token_path)
            
            if creds and creds.valid:
                print("✅ Existing token is valid!")
                return True
            elif creds and creds.expired and creds.refresh_token:
                print("🔄 Refreshing expired token...")
                creds.refresh(Request())
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
                print("✅ Token refreshed successfully!")
                return True
        
        # Start OAuth flow
        print("\n🔗 Starting OAuth2 authentication flow...")
        
        flow = Flow.from_client_secrets_file(
            creds_path,
            scopes=['https://www.googleapis.com/auth/calendar'],
            redirect_uri=redirect_uri
        )
        
        # Generate authorization URL
        auth_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        
        print(f"\n🔗 Please visit this URL to authorize the application:")
        print(f"{auth_url}")
        print(f"\n📋 After authorization, you will get a code. Please enter it below:")
        
        code = input('Enter the authorization code: ').strip()
        if not code:
            print("❌ No authorization code provided. Setup cancelled.")
            return False
        
        # Exchange code for token
        flow.fetch_token(code=code)
        creds = flow.credentials
        
        # Save token
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
        
        print("✅ Authentication successful! Token saved.")
        print("🎉 Google Calendar authentication is now set up!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during setup: {str(e)}")
        return False

if __name__ == "__main__":
    success = setup_calendar_auth()
    if success:
        print("\n🚀 You can now use the Google Calendar Agent!")
    else:
        print("\n❌ Setup failed. Please check the instructions above.") 