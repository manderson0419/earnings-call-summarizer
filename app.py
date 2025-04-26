from flask import Flask, request
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
import io
import openai
import fitz  # PyMuPDF
from googleapiclient.http import MediaIoBaseDownload
import json
import time
import tiktoken

app = Flask(__name__)

# Set your OpenAI API key from environment variables
openai.api_key = os.environ['OPENAI_API_KEY']

# Folder ID to watch
FOLDER_ID = '1VsWkYlSJSFWHRK6u66qKhUn9xqajMPd6'

# Initialize global debounce
is_summarizing = False

# Function to get Drive service
def get_drive_service():
    credentials_info = json.loads(os.environ['GOOGLE_CREDENTIALS_JSON'])
    creds = service_account.Credentials.from_service_account_info(
        credentials_info,
        scopes=['https://www.googleapis.com/auth/drive']
    )
    return build('drive', 'v3', credentials=creds)

# Function to count tokens using tiktoken
def count_tokens(text, model='gpt-4o'):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

# Summarize function
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

    # Chunk by ~9000 tokens
    CHUNK_TOKEN_SIZE = 9000
    tokens = count_tokens(text)
    print(f"Total tokens in text: {tokens}")

    chunks = []
    current_chunk = ""
    for paragraph in text.split('\n'):
        if count_tokens(current_chunk + paragraph) > CHUNK_TOKEN_SIZE:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            current_chunk += paragraph + "\n"
    if current_chunk:
        chunks.append(current_chunk)

    print(f"Splitting into {len(chunks)} chunk(s) for summarization...")

    combined_summary = ""

    for idx, chunk in enumerate(chunks):
        print(f"Summarizing chunk {idx+1}/{len(chunks)}...")
        retries = 0
        while retries < 5:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": chunk}
                    ],
                    temperature=0.3,
                    timeout=60
                )
                chunk_summary = response.choices[0].message.content
                combined_summary += f"\n\n{chunk_summary}"
                print(f"Finished chunk {idx+1}, sleeping 25s to avoid rate limit...")
                time.sleep(25)
                break  # exit retry loop if success
            except openai.RateLimitError as e:
                retry_after = int(e.response.headers.get('Retry-After', 45))
                retries += 1
                print(f"Rate limit error (retry {retries}). Waiting {retry_after} seconds...")
                time.sleep(retry_after)
            except Exception as e:
                retries += 1
                print(f"Error summarizing chunk {idx+1}: {e}. Retrying...")
                time.sleep(30)

    return combined_summary.strip()

# Extract text from PDF
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Webhook endpoint
@app.route('/webhook', methods=['POST'])
def webhook():
    global is_summarizing

    if is_summarizing:
        print("Another summarization is already in progress. Skipping this webhook.")
        return '', 200

    print("Received a webhook notification.")
    print("Request headers:", request.headers)

    is_summarizing = True
    try:
        service = get_drive_service()

        results = service.files().list(
            q=f"'{FOLDER_ID}' in parents and trashed = false",
            fields="files(id, name, createdTime, modifiedTime)",
            orderBy="modifiedTime desc",
            pageSize=1
        ).execute()

        items = results.get('files', [])

        if not items:
            print('Warning: No files found in the folder.')
        else:
            file = items[0]
            file_id = file['id']
            file_name = file['name']
            print(f"Newest file: {file_name} (ID: {file_id})")

            # Download the file
            request_drive = service.files().get_media(fileId=file_id)
            fh = io.FileIO(file_name, 'wb')
            downloader = MediaIoBaseDownload(fh, request_drive)

            done = False
            while not done:
                status, done = downloader.next_chunk()

            print(f"Downloaded file: {file_name}")

            if file_name.endswith('.pdf'):
                print("Detected PDF, extracting text...")
                file_contents = extract_text_from_pdf(file_name)
            else:
                print("Detected text file, reading contents...")
                with open(file_name, 'r', encoding='utf-8') as f:
                    file_contents = f.read()

            print(f"Text length: {len(file_contents)} characters")

            # Summarize
            print("Summarizing file...")
            summary = summarize_text(file_contents)

            print("\nSUMMARY (Ready for Slack):")
            print(summary)

    except Exception as e:
        print(f"Error during summarization flow: {e}")

    finally:
        is_summarizing = False

    return '', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
