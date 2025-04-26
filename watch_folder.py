from google.oauth2 import service_account
from googleapiclient.discovery import build
import uuid

# Path to your service account key file
SERVICE_ACCOUNT_FILE = '/Users/michaelaanderson/Downloads/tidy-centaur-457916-h9-2230742befee.json'

# ID of the folder you want to watch
FOLDER_ID = '1VsWkYlSJSFWHRK6u66qKhUn9xqajMPd6'

# Your webhook URL (your ngrok HTTPS URL)
WEBHOOK_URL = 'https://earnings-call-summarizer-production.up.railway.app/webhook'

def watch_folder():
    # Authenticate
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=['https://www.googleapis.com/auth/drive']
    )

    service = build('drive', 'v3', credentials=creds)

    # Set up the watch request
    request_body = {
        'id': str(uuid.uuid4()),  # Random unique ID for the channel
        'type': 'web_hook',
        'address': WEBHOOK_URL
    }

    # Call the Drive API to watch the folder
    response = service.files().watch(fileId=FOLDER_ID, body=request_body).execute()

    print('✅ Successfully set up watch on folder!')
    print('Response from Google:')
    print(response)

if __name__ == '__main__':
    watch_folder()

