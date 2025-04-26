from flask import Flask, request
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
import io
import openai
import fitz  # PyMuPDF
import tiktoken
import time
import json
from googleapiclient.http import MediaIoBaseDownload

app = Flask(__name__)

openai.api_key = os.environ['OPENAI_API_KEY']
FOLDER_ID = '1VsWkYlSJSFWHRK6u66qKhUn9xqajMPd6'

is_summarizing = False

TOKEN_LIMIT_PER_REQUEST = 9000  # safe chunk size

# Set up tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text):
    return len(tokenizer.encode(text))

def split_into_token_chunks(text, max_tokens=TOKEN_LIMIT_PER_REQUEST):
    words = text.split()
    chunks = []
    current_chunk = []
    current_token_count = 0

    for word in words:
        token_count = len(tokenizer.encode(word + ' '))
        if current_token_count + token_count > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_token_count = token_count
        else:
            current_chunk.append(word)
            current_token_count += token_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

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

    chunks = split_into_token_chunks(text, max_tokens=TOKEN_LIMIT_PER_REQUEST)
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
                    temperature=0.3
                )
                chunk_summary = response.choices[0].message.content
                combined_summary += f"\n\n{chunk_summary}"
                break  # Success, break retry loop
            except openai.RateLimitError as e:
                wait_time = 45  # default wait time
                if hasattr(e, 'response') and e.response and 'Retry-After' in e.response.headers:
                    wait_time = int(e.response.headers['Retry-After'])
                print(f"Rate limit error (retry {retries+1}). Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            except Exception as e:
                print(f"Warning: Error summarizing chunk {idx+1}: {e}")
                combined_summary += f"\n\nWarning: Error summarizing part {idx+1}: {e}"
                break

    return combined_summary.strip()

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@app.route('/webhook', methods=['POST'])
def webhook():
    global is_summarizing
    if is_summarizing:
        print("Another summarization is already in progress. Skipping this webhook.")
        return '', 200

    print("Received a webhook notification.")
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
        print('Warning: No files found in the folder.')
        return '', 200

    file = items[0]
    file_id = file['id']
    file_name = file['name']
    print(f"Newest file: {file_name} (ID: {file_id})")

    try:
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

        is_summarizing = True
        print("Summarizing file...")
        summary = summarize_text(file_contents)
        is_summarizing = False

        print("\nSUMMARY (Ready for Slack):")
        print(summary)

    except Exception as e:
        is_summarizing = False
        print(f"Warning: Could not process file: {e}")

    return '', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

