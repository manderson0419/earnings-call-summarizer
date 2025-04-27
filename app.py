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
import re
import requests
from googleapiclient.http import MediaIoBaseDownload

app = Flask(__name__)

openai.api_key = os.environ['OPENAI_API_KEY']
SLACK_WEBHOOK_URL = os.environ['SLACK_WEBHOOK_URL']
FOLDER_ID = '1VsWkYlSJSFWHRK6u66qKhUn9xqajMPd6'

is_summarizing = False
TOKEN_LIMIT_PER_REQUEST = 9000

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

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)  # Collapse multiple newlines
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    text = text.replace(' \n', '\n').replace('\n ', '\n')
    return text.strip()

def format_for_slack(summary_text):
    formatted = summary_text
    formatted = formatted.replace("Key Financial Highlights:", "\n:moneybag: *Financial Highlights:*")
    formatted = formatted.replace("Key Operational Highlights:", "\n:factory: *Operational Highlights:*")
    formatted = formatted.replace("Forward Guidance:", "\n:crystal_ball: *Forward Guidance:*")
    formatted = formatted.replace("Sentiment Analysis:", "\n:bar_chart: *Sentiment Analysis:*")
    formatted = formatted.replace("Executive Summary:", "\n:small_blue_diamond: *Executive Summary:*")
    formatted = formatted.strip()
    formatted += "\n---"
    return formatted

def get_drive_service():
    credentials_info = json.loads(os.environ['GOOGLE_CREDENTIALS_JSON'])
    creds = service_account.Credentials.from_service_account_info(
        credentials_info,
        scopes=['https://www.googleapis.com/auth/drive']
    )
    return build('drive', 'v3', credentials=creds)

def send_to_slack(message):
    payload = {"text": message}
    response = requests.post(SLACK_WEBHOOK_URL, json=payload)
    if response.status_code != 200:
        print(f"Warning: Slack API returned {response.status_code}: {response.text}")
    else:
        print("Slack message sent successfully!")

def summarize_text(text):
    system_prompt = (
        "You are a professional financial analyst AI assistant. "
        "You will be provided an earnings call transcript or a section of one. "
        "Extract:
        - Financial Highlights
        - Operational Highlights
        - Forward Guidance
        - Sentiment Analysis
        - Executive Summary

        Use clear labels, Slack-ready markdown, and make it very succinct and professional."
    )

    client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    chunks = split_into_token_chunks(text)
    print(f"Splitting into {len(chunks)} chunk(s) for summarization...")

    partial_summaries = []

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
                chunk_summary = response.choices[0].message.content.strip()
                partial_summaries.append(chunk_summary)
                break
            except openai.RateLimitError as e:
                wait_time = 120
                if hasattr(e, 'response') and e.response and 'Retry-After' in e.response.headers:
                    wait_time = int(e.response.headers['Retry-After'])
                print(f"Rate limit error (retry {retries+1}). Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            except Exception as e:
                print(f"Warning: Error summarizing chunk {idx+1}: {e}")
                partial_summaries.append(f"Error summarizing part {idx+1}: {e}")
                break

    print("Combining all chunk summaries into final synthesis...")

    combined_summary_text = "\n\n".join(partial_summaries)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial analyst assistant. Combine the following partial summaries into a final clean, Slack-ready earnings call summary with sections, emojis, and professional formatting."},
                {"role": "user", "content": combined_summary_text}
            ],
            temperature=0.2
        )
        final_summary = response.choices[0].message.content.strip()
        return final_summary
    except Exception as e:
        print(f"Warning: Error during final summarization: {e}")
        return combined_summary_text

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

        file_contents = clean_text(file_contents)
        print(f"Text length after cleaning: {len(file_contents)} characters")

        is_summarizing = True
        print("Summarizing file...")
        summary = summarize_text(file_contents)
        formatted_summary = format_for_slack(summary)

        send_to_slack(formatted_summary)
        is_summarizing = False

    except Exception as e:
        is_summarizing = False
        print(f"Warning: Could not process file: {e}")

    return '', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
