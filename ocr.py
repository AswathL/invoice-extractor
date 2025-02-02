import streamlit as st
import easyocr
import requests
import pandas as pd
import json
from pdf2image import convert_from_bytes
from PIL import Image
import base64

# Initialize EasyOCR reader
reader = easyocr.Reader(["en"])

# Function to extract text from image
def extract_text_from_image(image):
    result = reader.readtext(image, detail=0)
    return "\n".join(result)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pages = convert_from_bytes(pdf_file.read())
    return "\n".join([extract_text_from_image(page) for page in pages])

# Function to process extracted text with Ollama
def process_text_with_ollama(text):
    url = "http://localhost:11434/api/generate"
    prompt = f"""Extract key invoice details in JSON format:
    - Invoice Number
    - Invoice Date
    - Vendor Name
    - Items: List of {{'Description': '', 'Quantity': '', 'Unit Price': '', 'Amount': ''}}
    - Total Amount before Tax
    - Tax Amount
    - Discounts
    - Grand Total

    Extract from the following text:
    {text}
    """
    
    response = requests.post(url, json={"model": "mistral", "prompt": prompt, "stream": False})
    if response.status_code == 200:
        return response.json().get("response", "{}")
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Function to parse JSON data
def parse_invoice_data(data):
    try:
        invoice_data = json.loads(data)
        if isinstance(invoice_data, dict):
            invoice_list = [invoice_data]
        elif isinstance(invoice_data, list):
            invoice_list = invoice_data
        else:
            raise ValueError("Unexpected invoice data format")

        all_keys = set()
        for invoice in invoice_list:
            all_keys.update(invoice.keys())
            for item in invoice.get("Items", []):
                all_keys.update(item.keys())

        all_keys.discard("Items")

        structured_data = []
        for invoice in invoice_list:
            base_data = {k: invoice.get(k, "N/A") for k in all_keys}
            items = invoice.get("Items", [])
            if isinstance(items, list):
                for item in items:
                    row = {**base_data, **{k: item.get(k, "N/A") for k in item}}
                    structured_data.append(row)
            else:
                structured_data.append(base_data)

        return structured_data, list(all_keys)
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON: {e}")
        return [], []
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return [], []

# Streamlit UI
def main():
    st.title("Invoice Data Extractor")
    st.write("Upload an invoice (PDF or image) to extract structured data.")

    uploaded_file = st.file_uploader("Upload Invoice", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.write(f"File uploaded: {uploaded_file.name}")

        extracted_text = extract_text_from_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else extract_text_from_image(Image.open(uploaded_file))

        st.subheader("Extracted Text")
        st.write(extracted_text)

        if st.button("Process Invoice"):
            with st.spinner("Processing invoice..."):
                structured_data = process_text_with_ollama(extracted_text)
                st.subheader("Structured Data")
                st.json(structured_data)

                parsed_data, headers = parse_invoice_data(structured_data)
                if parsed_data:
                    df = pd.DataFrame(parsed_data, columns=headers)
                    st.subheader("Structured Table")
                    st.dataframe(df)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download CSV", csv, "invoice_data.csv", "text/csv")
                else:
                    st.warning("No structured data found.")

if __name__ == "__main__":
    main()
