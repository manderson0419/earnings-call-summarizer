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

def clean_unicode(text):
    return text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.replace(' \n', '\n').replace('\n ', '\n')
    return text.strip()

def get_drive_service():
    credentials_info = json.loads(os.environ['GOOGLE_CREDENTIALS_JSON'])
    creds = service_account.Credentials.from_service_account_info(
        credentials_info,
        scopes=['https://www.googleapis.com/auth/drive']
    )
    return build('drive', 'v3', credentials=creds)

def format_for_slack(summary_text):
    # Replace "### Heading" patterns
    formatted = summary_text

    # Remove markdown '###' headers if they exist and replace them with emojis + bold
    formatted = re.sub(r'###\s*Financial Highlights', "📊 *Financial Highlights*", formatted)
    formatted = re.sub(r'###\s*Operational Highlights', "🏢 *Operational Highlights*", formatted)
    formatted = re.sub(r'###\s*Forward Guidance', "🔮 *Forward Guidance*", formatted)
    formatted = re.sub(r'###\s*Sentiment Analysis', "📈 *Sentiment Analysis*", formatted)

    # If any "**text**" is still left, replace it with "*text*"
    formatted = re.sub(r'\*\*(.*?)\*\*', r'*\1*', formatted)

    # Replace leading hyphens "-" with a round bullet "•"
    formatted = re.sub(r'^\s*-\s*', "• ", formatted, flags=re.MULTILINE)

    # Collapse multiple newlines into single
    formatted = re.sub(r'\n\n+', '\n', formatted)

    return formatted.strip()

def send_to_slack(message_text):
    payload = {
        "text": message_text
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(SLACK_WEBHOOK_URL, json=payload, headers=headers)
    if response.status_code != 200:
        print(f"Failed to send message to Slack: {response.status_code}, {response.text}")
    else:
        print("Successfully sent message to Slack!")

def summarize_chunks(chunks):
    system_prompt = (
        "You are a professional financial analyst AI assistant. "
        "Summarize each chunk of the earnings call into structured bullet points under these sections: "
        "Financial Highlights, Operational Highlights, Forward Guidance, Sentiment Analysis. "
        "Split sentiment analysis between internal company sentiment (executive comments) and external sentiment (analyst questions). "
        "Use clear bullet points, avoid technical jargon unless stated. "
        "Do NOT create an 'Executive Summary'. "
        "Format cleanly for Slack posts, no extra asterisks inside the bullets."
    )
    client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

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
                break
            except openai.RateLimitError as e:
                wait_time = 45
                if hasattr(e, 'response') and e.response and 'Retry-After' in e.response.headers:
                    wait_time = min(int(e.response.headers['Retry-After']), 120)
                print(f"Rate limit error (retry {retries+1}). Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            except Exception as e:
                print(f"Warning: Error summarizing chunk {idx+1}: {e}")
                break

    return combined_summary.strip()

def summarize_combined_summary(combined_summary_text):
    system_prompt = (
        "You have received multiple partial summaries of an earnings call. "
        "Combine them into ONE final clean Slack-ready summary under the sections: "
        "Financial Highlights, Operational Highlights, Forward Guidance, Sentiment Analysis. "
        "Use bullet points, clean formatting, and Slack-ready formatting with emojis and bold section titles. "
        "Do NOT create an Executive Summary."
    )
    client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    retries = 0
    while retries < 5:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": combined_summary_text}
                ],
                temperature=0.2
            )
            final_summary = response.choices[0].message.content
            return final_summary.strip()
        except openai.RateLimitError as e:
            wait_time = 45
            if hasattr(e, 'response') and e.response and 'Retry-After' in e.response.headers:
                wait_time = min(int(e.response.headers['Retry-After']), 120)
            print(f"Rate limit error (retry {retries+1}). Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
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

        file_contents = clean_unicode(file_contents)
        file_contents = clean_text(file_contents)
        print(f"Text length after cleaning: {len(file_contents)} characters")

        is_summarizing = True
        print("Summarizing file...")

        chunks = split_into_token_chunks(file_contents)
        combined_summary = summarize_chunks(chunks)
        final_summary = summarize_combined_summary(combined_summary)

        slack_message = format_for_slack(final_summary)
        send_to_slack(slack_message)

        is_summarizing = False

    except Exception as e:
        is_summarizing = False
        print(f"Warning: Could not process file: {e}")

    return '', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

