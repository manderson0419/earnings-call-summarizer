from flask import Flask, request
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
import io
import openai
import fitz  # PyMuPDF
from googleapiclient.http import MediaIoBaseDownload
import json

app = Flask(__name__)

# Set your OpenAI API key from environment variables
openai.api_key = os.environ['OPENAI_API_KEY']

# Folder ID to watch
FOLDER_ID = '1VsWkYlSJSFWHRK6u66qKhUn9xqajMPd6'

def get_drive_service():
    credentials_info = json.loads(os.environ['GOOGLE_CREDENTIALS_JSON'])
    creds = service_account.Credentials.from_service_account_info(
        credentials_info,
        scopes=['https://www.googleapis.com/auth/drive']
    )
    return build('drive', 'v3', credentials=creds)

def summarize_text(text):
    system_prompt = (
        "You are a professional financial analyst AI assistant. "
        "You take earnings call transcripts as input, process the content, and generate:\n"
        "- Key Financial Highlights (bullet points)\n"
        "- Key Operational Highlights (bullet points)\n"
        "- Forward Guidance (bullet points)\n"
        "- Sentiment Analysis (bullet points)\n"
        "- A concise executive Summary\n\n"
        "Format the output cleanly. Financial summaries must:\n"
        "- Be in 2 to 6 concise bullet points under 20 words each\n"
        "- Highlight Revenue, EPS, and notable financial announcements when available\n"
        "- Use clear labels (e.g., 'Revenue:', 'EPS:')\n"
        "- Avoid assuming missing data; say 'Data not available' if needed\n"
        "- Prepare the output as a Slack-compatible message (markdown-friendly)\n"
        "- Maintain a professional tone for finance teams and executives\n"
        "- Handle inconsistent transcript formats gracefully\n"
        "- Clearly distinguish between quantitative data and qualitative sentiment\n\n"
        "Process the following transcript and output accordingly:\n\n"
    )

    client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            timeout=60  # set a timeout to avoid hanging
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        print(f"\u26a0\ufe0f OpenAI Summarization Error: {e}")
        return "\u26a0\ufe0f Failed to summarize due to error."

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@app.route('/webhook', methods=['POST'])
def webhook():
    print("=== Received a webhook notification ===")
    print("Request headers:", request.headers)

    service = get_drive_service()

    results = service.files().list(
        q=f"'{FOLDER_ID}' in parents and trashed = false",
        fields="files(id, name, createdTime, modifiedTime)",
        orderBy="modifiedTime desc",
        pageSize=1
    ).execute()

    items = results.get('files', [])

    if not items:
        print('\u26a0\ufe0f No files found in the folder.')
    else:
        file = items[0]
        file_id = file['id']
        file_name = file['name']
        print(f"\ud83d\udcc4 Newest file: {file_name} (ID: {file_id})")

        # Download the file
        request_drive = service.files().get_media(fileId=file_id)
        fh = io.FileIO(file_name, 'wb')
        downloader = MediaIoBaseDownload(fh, request_drive)

        done = False
        while done is False:
            status, done = downloader.next_chunk()

        print(f"\u2705 Downloaded file: {file_name}")

        try:
            if file_name.endswith('.pdf'):
                print("\ud83d\udcc4 Detected PDF, extracting text...")
                file_contents = extract_text_from_pdf(file_name)
            else:
                print("\ud83d\udcc4 Detected text file, reading contents...")
                with open(file_name, 'r', encoding='utf-8') as f:
                    file_contents = f.read()

            print(f"\ud83d\udd22 Text length: {len(file_contents)} characters")

            # Summarize the contents
            print("\u2728 Summarizing file...")
            summary = summarize_text(file_contents)

            # Print the summary
            print("\n\ud83d\udccb SUMMARY (Ready for Slack):")
            print(summary)
        except Exception as e:
            print(f"\u26a0\ufe0f Could not read or summarize the file: {e}")

    return '', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
