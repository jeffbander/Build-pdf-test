# main.py
import streamlit as st
import pdfplumber
from io import BytesIO
import re
import json
from typing import List, Dict
import html
import base64
import os

# Optional: Mistral AI integration
try:
    from mistralai import Mistral
    mistral_available = True
except ImportError:
    mistral_available = False

def compress_text(text: str) -> str:
    """Compress text by removing excessive whitespace and cleaning formatting."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return '\n'.join(line.strip() for line in text.split('\n')).strip()

def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 200) -> List[Dict]:
    """Split text into overlapping chunks optimized for AI processing."""
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        chunks.append({
            'chunk_id': len(chunks) + 1,
            'text': chunk_text,
            'word_count': len(chunk_words),
            'start_word': i,
            'end_word': min(i + chunk_size, len(words))
        })
    return chunks

def extract_metadata(text: str, page_num: int, tables: List[str], captions: List[str]) -> Dict:
    """Extract metadata and key information from page text, tables, and captions."""
    word_count = len(text.split())
    char_count = len(text)
    lines = text.split('\n')
    headings = [line.strip() for line in lines if len(line.strip()) < 100 and line.strip().istitle()]
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    dates = re.findall(r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b', text)
    return {
        'page': page_num,
        'word_count': word_count,
        'char_count': char_count,
        'headings': headings[:5],
        'numbers_found': len(numbers),
        'dates_found': len(dates),
        'has_tables': bool(tables),
        'has_captions': bool(captions),
        'table_count': len(tables),
        'caption_count': len(captions)
    }

def generate_html_output(text: str, chunks_data: List[Dict], metadata: List[Dict], filename: str) -> str:
    """Generate HTML output optimized for AI agent searching."""
    all_headings = [h for meta in metadata for h in meta.get('headings', [])]
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document: {html.escape(filename)}</title>
    <meta name="description" content="AI-optimized document extract from {html.escape(filename)}">
    <meta name="keywords" content="{', '.join(html.escape(h) for h in all_headings[:10])}">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f9f9f9; }}
        .document-header {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .page-section {{ background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .page-header {{ background: #007bff; color: white; padding: 10px 15px; margin: -20px -20px 15px -20px; border-radius: 8px 8px 0 0; font-weight: bold; }}
        .chunk-section {{ border-left: 4px solid #28a745; padding-left: 15px; margin: 15px 0; background: #f8f9fa; padding: 15px; border-radius: 0 8px 8px 0; }}
        .metadata {{ background: #e9ecef; padding: 10px; border-radius: 5px; font-size: 0.9em; margin-top: 10px; }}
        .searchable-content {{ white-space: pre-wrap; font-family: Georgia, serif; line-height: 1.8; }}
        .toc {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .toc ul {{ list-style-type: none; padding-left: 0; }}
        .toc li {{ padding: 5px 0; }}
        .toc a {{ text-decoration: none; color: #007bff; }}
        .toc a:hover {{ text-decoration: underline; }}
        .highlight {{ background-color: yellow; }}
        .keyword {{ font-weight: bold; color: #dc3545; }}
        .table-content {{ background: #f1f3f5; padding: 10px; border-radius: 5px; margin-top: 10px; }}
        .caption-content {{ font-style: italic; margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="document-header">
        <h1>📄 {html.escape(filename)}</h1>
        <p><strong>Total Pages:</strong> {len(metadata)} | <strong>Total Chunks:</strong> {len(chunks_data)} | <strong>Processing Date:</strong> {json.dumps({})[1:-1]}</p>
        <p><em>AI-optimized extract with text, tables, and figure captions.</em></p>
    </div>
"""
    if all_headings:
        html_content += """
    <div class="toc">
        <h2>📋 Table of Contents</h2>
        <ul>
"""
        for i, heading in enumerate(all_headings[:20]):
            anchor = f"heading-{i}"
            html_content += f'            <li><a href="#{anchor}">{html.escape(heading)}</a></li>\n'
        html_content += """        </ul>
    </div>
"""
    pages = text.split('--- Page ')
    heading_counter = 0
    for i, page_content in enumerate(pages[1:], 1):
        lines = page_content.split('\n', 1)
        page_num = lines[0].split(' ---')[0]
        page_text = lines[1] if len(lines) > 1 else ""
        page_meta = next((m for m in metadata if m['page'] == int(page_num)), {})
        html_content += f"""
    <div class="page-section" id="page-{page_num}">
        <div class="page-header">
            📄 Page {page_num}
        </div>
"""
        if page_meta:
            html_content += f"""
        <div class="metadata">
            <strong>Page Stats:</strong> {page_meta.get('word_count', 0)} words, {page_meta.get('char_count', 0)} characters
            {f" | <strong>Tables:</strong> {page_meta.get('table_count', 0)}" if page_meta.get('has_tables') else ""}
            {f" | <strong>Captions:</strong> {page_meta.get('caption_count', 0)}" if page_meta.get('has_captions') else ""}
            {f" | <strong>Dates Found:</strong> {page_meta.get('dates_found', 0)}" if page_meta.get('dates_found', 0) > 0 else ""}
        </div>
"""
        if page_meta.get('headings'):
            html_content += "        <div class=\"metadata\"><strong>Key Topics:</strong> "
            for j, heading in enumerate(page_meta['headings']):
                anchor = f"heading-{heading_counter}"
                html_content += f'<span id="{anchor}" class="keyword">{html.escape(heading)}</span>'
                if j < len(page_meta['headings']) - 1:
                    html_content += ", "
                heading_counter += 1
            html_content += "</div>\n"
        if page_meta.get('tables'):
            html_content += "        <div class=\"table-content\"><strong>Tables:</strong><br>"
            for table in page_meta['tables']:
                html_content += f"<pre>{html.escape(table)}</pre>"
            html_content += "</div>\n"
        if page_meta.get('figure_captions'):
            html_content += "        <div class=\"caption-content\"><strong>Figure Captions:</strong><br>"
            for caption in page_meta['figure_captions']:
                html_content += f"{html.escape(caption)}<br>"
            html_content += "</div>\n"
        html_content += f"""
        <div class="searchable-content">{html.escape(page_text)}</div>
    </div>
"""
    if chunks_data:
        html_content += """
    <div class="document-header">
        <h2>🧩 AI-Optimized Content Chunks</h2>
        <p><em>Chunks optimized for AI processing with overlap for context preservation.</em></p>
    </div>
"""
        for chunk in chunks_data:
            html_content += f"""
    <div class="chunk-section" id="chunk-{chunk['chunk_id']}">
        <strong>Chunk {chunk['chunk_id']}</strong> | Words: {chunk['word_count']} | Range: {chunk['start_word']}-{chunk['end_word']}
        <div class="searchable-content">{html.escape(chunk['text'])}</div>
    </div>
"""
    html_content += "</body></html>"
    return html_content

st.title("AI-Optimized PDF Processor for Complex Documents")

st.markdown("""
This app processes complex PDFs (up to 300 pages, 50 MB) with tables, figures, and varied layouts, optimized for AI search. It includes text compression, semantic chunking, metadata extraction, table extraction, and figure caption detection. Now with optional Mistral OCR integration for scanned or low-quality PDFs (e.g., faxes), using mistral-ocr-latest for full structure recovery including tables, figures, and layouts.

**Note**: Mistral OCR requires an API key from Mistral AI. Scanned PDFs benefit from OCR; text-based PDFs use pdfplumber for faster processing.
""")

# Processing options
st.sidebar.header("Processing Options")
compress_text_option = st.sidebar.checkbox("Compress text (remove excess whitespace)", value=True)
create_chunks = st.sidebar.checkbox("Create AI-optimized chunks", value=True)
chunk_size = st.sidebar.slider("Chunk size (words)", 1000, 8000, 4000)
extract_metadata_option = st.sidebar.checkbox("Extract metadata", value=True)
extract_tables = st.sidebar.checkbox("Extract tables", value=True)
extract_captions = st.sidebar.checkbox("Extract figure captions", value=True)
generate_html = st.sidebar.checkbox("Generate HTML for AI agent search", value=True)

use_mistral_ocr = st.sidebar.checkbox("Use Mistral OCR for scanned/low-quality PDFs", value=False)
if use_mistral_ocr:
    if not mistral_available:
        st.sidebar.error("Mistral AI SDK not installed. Add 'mistralai' to requirements.txt and run pip install.")
    mistral_api_key = "R3mk2Mw4jrPtXbcafdsBDNf17RKv2F2x"  # Built-in API key
    st.sidebar.success("✅ Mistral API key configured")
    include_annotations = st.sidebar.checkbox("Include annotations (bounding boxes for tables/figures)", value=True)
else:
    mistral_api_key = None

uploaded_file = st.file_uploader("Upload PDF", type="pdf", help="Maximum file size: 50 MB")

if uploaded_file:
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
    st.info(f"Uploaded file size: {file_size:.2f} MB")
    
    # File size validation
    if file_size > 50:
        st.error(f"File too large ({file_size:.2f} MB). Please upload a PDF smaller than 50 MB.")
        st.stop()
    elif file_size > 25:
        st.warning(f"Large file detected ({file_size:.2f} MB). Processing may take longer and could fail due to memory limits.")

    if st.button("Process PDF"):
        with st.spinner("Opening PDF..."):
            try:
                pdf_bytes = uploaded_file.read()
                text = ""
                all_metadata = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                num_pages = 0

                if use_mistral_ocr and mistral_api_key and mistral_available:
                    # Use Mistral OCR
                    os.environ["MISTRAL_API_KEY"] = mistral_api_key
                    client = Mistral(api_key=mistral_api_key)
                    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                    document = {
                        "type": "document_url",
                        "document_url": f"data:application/pdf;base64,{base64_pdf}"
                    }
                    with st.spinner("Processing with Mistral OCR..."):
                        ocr_response = client.ocr.process(
                            model="mistral-ocr-latest",
                            document=document,
                            include_image_base64=True
                        )
                    # Process Mistral response
                    num_pages = len(ocr_response.pages)
                    for i, page in enumerate(ocr_response.pages):
                        page_num = page.index
                        page_text = page.markdown  # Markdown preserves structure
                        page_data = {"page": page_num, "text": page_text, "tables": [], "figure_captions": []}
                        # Extract images/figures with bboxes
                        for img in page.images:
                            caption = f"Figure ID: {img.id} at ({img.top_left_x}, {img.top_left_y}) - ({img.bottom_right_x}, {img.bottom_right_y})"
                            page_data["figure_captions"].append(caption)
                        # For tables, since markdown includes them, parse if needed
                        # Assuming markdown has tables, extract via regex or keep as is
                        if extract_tables:
                            # Simple table detection from markdown
                            tables = re.findall(r'\|.*\|(?:\n\|.*\|)+', page_text)
                            page_data["tables"] = tables
                        if compress_text_option:
                            page_data["text"] = compress_text(page_data["text"])
                        if extract_metadata_option:
                            metadata = extract_metadata(page_data["text"], page_num, page_data["tables"], page_data["figure_captions"])
                            metadata["tables"] = page_data["tables"]
                            metadata["figure_captions"] = page_data["figure_captions"]
                            all_metadata.append(metadata)
                        text += f"--- Page {page_num} ---\n{page_data['text']}\n"
                        if page_data["tables"]:
                            text += "Tables:\n" + "\n\n".join(page_data["tables"]) + "\n"
                        if page_data["figure_captions"]:
                            text += "Figure Captions:\n" + "\n".join(page_data["figure_captions"]) + "\n"
                        progress = (i + 1) / num_pages
                        progress_bar.progress(progress)
                        status_text.text(f"Processing page {i + 1} of {num_pages} ({progress * 100:.1f}%)")
                else:
                    # Use pdfplumber for text-based PDFs
                    pdf = pdfplumber.open(BytesIO(pdf_bytes))
                    num_pages = len(pdf.pages)
                    for i, page in enumerate(pdf.pages):
                        page_data = {"page": page.page_number, "text": "", "tables": [], "figure_captions": []}
                        page_text = page.extract_text(layout=True)
                        if page_text:
                            if compress_text_option:
                                page_text = compress_text(page_text)
                            page_data["text"] = page_text
                        else:
                            page_data["text"] = "(No text extracted)"
                        if extract_tables:
                            try:
                                tables = page.extract_tables()
                                for table in tables:
                                    table_str = "\n".join(["\t".join(cell or "" for cell in row) for row in table if row])
                                    page_data["tables"].append(table_str)
                            except Exception as table_error:
                                page_data["tables"].append(f"(Table extraction error: {str(table_error)})")
                        if extract_captions:
                            try:
                                captions = []
                                for obj in page.objects.get("char", []):
                                    if "text" in obj and any(keyword in obj["text"].lower() for keyword in ["figure", "fig.", "caption"]):
                                        captions.append(obj["text"])
                                page_data["figure_captions"] = captions if captions else ["(No captions detected)"]
                            except:
                                page_data["figure_captions"] = ["(No captions detected)"]
                        if extract_metadata_option:
                            metadata = extract_metadata(page_data["text"], page.page_number, page_data["tables"], page_data["figure_captions"])
                            metadata["tables"] = page_data["tables"]
                            metadata["figure_captions"] = page_data["figure_captions"]
                            all_metadata.append(metadata)
                        text += f"--- Page {page.page_number} ---\n{page_data['text']}\n"
                        if page_data["tables"]:
                            text += "Tables:\n" + "\n\n".join(page_data["tables"]) + "\n"
                        if page_data["figure_captions"]:
                            text += "Figure Captions:\n" + "\n".join(page_data["figure_captions"]) + "\n"
                        progress = (i + 1) / num_pages
                        progress_bar.progress(progress)
                        status_text.text(f"Processing page {i + 1} of {num_pages} ({progress * 100:.1f}%)")
                    pdf.close()

                original_size = len(pdf_bytes)
                compressed_size = len(text.encode('utf-8'))
                compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

                st.success("PDF processing complete!")
                st.info(f"Compression: {compression_ratio:.1f}% reduction | Text size: {compressed_size / 1024:.1f} KB")

                chunks_data = []
                if create_chunks:
                    with st.spinner("Creating AI-optimized chunks..."):
                        chunks_data = chunk_text(text, chunk_size)
                    st.info(f"Created {len(chunks_data)} chunks for AI processing")

                st.text_area("Extracted Content Preview (first 5000 characters)", text[:5000], height=300)

                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.download_button(
                        label="📄 Download Text",
                        data=text,
                        file_name=f"{uploaded_file.name}_extracted.txt",
                        mime="text/plain"
                    )
                with col2:
                    if chunks_data:
                        st.download_button(
                            label="🧩 Download Chunks",
                            data=json.dumps(chunks_data, indent=2),
                            file_name=f"{uploaded_file.name}_chunks.json",
                            mime="application/json"
                        )
                with col3:
                    if all_metadata:
                        st.download_button(
                            label="📊 Download Metadata",
                            data=json.dumps(all_metadata, indent=2),
                            file_name=f"{uploaded_file.name}_metadata.json",
                            mime="application/json"
                        )
                with col4:
                    if generate_html:
                        html_output = generate_html_output(text, chunks_data, all_metadata, uploaded_file.name)
                        st.download_button(
                            label="🌐 Download HTML",
                            data=html_output,
                            file_name=f"{uploaded_file.name}_search_optimized.html",
                            mime="text/html"
                        )
                with col5:
                    if chunks_data or all_metadata:
                        ai_ready_data = {
                            "document_info": {
                                "filename": uploaded_file.name,
                                "total_pages": num_pages,
                                "total_chunks": len(chunks_data),
                                "compression_ratio": f"{compression_ratio:.1f}%"
                            },
                            "chunks": chunks_data,
                            "metadata": all_metadata
                        }
                        st.download_button(
                            label="🤖 Download AI-Ready Format",
                            data=json.dumps(ai_ready_data, indent=2),
                            file_name=f"{uploaded_file.name}_ai_ready.json",
                            mime="application/json"
                        )

            except Exception as e:
                error_msg = str(e)
                if "413" in error_msg or "Request Entity Too Large" in error_msg or "Payload Too Large" in error_msg:
                    st.error("File too large for upload. Please use a smaller PDF (under 25 MB recommended for Replit).")
                elif "memory" in error_msg.lower() or "out of memory" in error_msg.lower():
                    st.error("Out of memory. Try a smaller file or enable text compression to reduce memory usage.")
                else:
                    st.error(f"Error processing PDF: {error_msg}")
                
                st.info("💡 Tips for large files:")
                st.info("• Use PDFs under 25 MB for best performance")
                st.info("• Enable text compression to reduce memory usage")
                st.info("• Split large documents into smaller sections")
                st.info("• For Mistral OCR, ensure API key is valid and file is under 50 MB")

else:
    st.warning("Please upload a PDF file to begin.")