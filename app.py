import streamlit as st
import PyPDF2
import docx
import anthropic
import openai
import google.generativeai as genai
import textwrap
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
import io
import re
from typing import List, Dict, Any
import time
import tiktoken

# Page configuration
st.set_page_config(
    page_title="Medical Study Notes Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e40af;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #374151;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        background-color: #dcfce7;
        border-left: 4px solid #16a34a;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fef3c7;
        border-left: 4px solid #d97706;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #dbeafe;
        border-left: 4px solid #2563eb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class TextProcessor:
    """Handles text preprocessing, chunking, and splitting"""
    
    def __init__(self):
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            self.encoding = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback estimation
            return len(text.split()) * 1.3
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = text.replace('', '')  # Remove replacement characters
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
        
        return text.strip()
    
    def recursive_text_split(self, text: str, max_chunk_size: int = 3000, overlap: int = 200) -> List[str]:
        """Recursively split text into chunks"""
        if self.count_tokens(text) <= max_chunk_size:
            return [text]
        
        # Try splitting by sections first
        sections = re.split(r'\n(?=[A-Z][^\n]*:|\d+\.|\w+\s+\d+)', text)
        
        if len(sections) > 1:
            chunks = []
            current_chunk = ""
            
            for section in sections:
                if self.count_tokens(current_chunk + section) <= max_chunk_size:
                    current_chunk += section + "\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = section + "\n"
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        
        # Fallback to paragraph splitting
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if self.count_tokens(current_chunk + para) <= max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If paragraph is too long, split by sentences
                if self.count_tokens(para) > max_chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    temp_chunk = ""
                    for sentence in sentences:
                        if self.count_tokens(temp_chunk + sentence) <= max_chunk_size:
                            temp_chunk += sentence + " "
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence + " "
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
                else:
                    chunks.append(para)
                
                current_chunk = ""
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

class LLMProcessor:
    """Handles different LLM integrations"""
    
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        try:
            if "claude" in self.model_name.lower():
                self.client = anthropic.Anthropic(api_key=self.api_key)
            elif "gpt" in self.model_name.lower():
                openai.api_key = self.api_key
                self.client = openai
            elif "gemini" in self.model_name.lower():
                genai.configure(api_key=self.api_key)
                self.client = genai
        except Exception as e:
            st.error(f"Error initializing {self.model_name}: {str(e)}")
    
    def generate_study_notes(self, text_chunk: str, content_type: str, field: str = "medicine") -> str:
        """Generate study notes using the specified LLM"""
        
        prompt = f"""Convert this {content_type} into fully detailed notes with the following rules:

• Do not summarize or omit any content
• Rewrite in structured, bullet-based format
• Preserve every point, explanation, and example
• Divide into logical sections and subsections
• Make it easily understandable for a postgraduate {field} student
• Use bold headings, indentation, and numbering where needed
• Keep bullet points as complete thoughts, not fragmented phrases

Additional requirements:
• Include all tables, figures, and case examples with full details
• Preserve all statistics, percentages, and numerical data
• Maintain all author names, study details, and clinical findings
• Keep all drug names, dosages, and monitoring requirements
• Include all diagnostic criteria and classification systems
• End at the conclusion section (exclude references/disclosures)

Format style:
• Use **bold** for main headings and key terms
• Use numbered lists for sequential information
• Use bullet points for related but non-sequential information
• Create clear hierarchy with sections (I, II, III) and subsections (A, B, C)

Text to convert:
{text_chunk}

Please provide comprehensive study notes following all the above guidelines."""

        try:
            if "claude" in self.model_name.lower():
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            elif "gpt" in self.model_name.lower():
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                    temperature=0.3
                )
                return response.choices[0].message.content
            
            elif "gemini" in self.model_name.lower():
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt)
                return response.text
            
        except Exception as e:
            st.error(f"Error generating notes with {self.model_name}: {str(e)}")
            return f"Error generating notes: {str(e)}"

class PDFGenerator:
    """Handles PDF generation for study notes"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=16,
            fontName='Helvetica-Bold',
            alignment=TA_LEFT,
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=14,
            fontName='Helvetica-Bold',
            alignment=TA_LEFT,
            spaceAfter=6,
            spaceBefore=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            fontName='Helvetica',
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            leftIndent=0
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBullet',
            parent=self.styles['Normal'],
            fontSize=11,
            fontName='Helvetica',
            alignment=TA_JUSTIFY,
            spaceAfter=3,
            leftIndent=20,
            bulletIndent=10
        ))
    
    def markdown_to_pdf_content(self, markdown_text: str) -> List:
        """Convert markdown text to PDF content"""
        content = []
        lines = markdown_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                content.append(Spacer(1, 6))
                continue
            
            # Main headings
            if line.startswith('# '):
                text = line[2:].strip()
                content.append(Paragraph(text, self.styles['CustomTitle']))
            
            # Sub headings
            elif line.startswith('## '):
                text = line[3:].strip()
                content.append(Paragraph(text, self.styles['CustomHeading']))
            
            # Bold text
            elif line.startswith('**') and line.endswith('**'):
                text = f"<b>{line[2:-2]}</b>"
                content.append(Paragraph(text, self.styles['CustomBody']))
            
            # Bullet points
            elif line.startswith('• ') or line.startswith('- '):
                text = line[2:].strip()
                content.append(Paragraph(f"• {text}", self.styles['CustomBullet']))
            
            # Numbered lists
            elif re.match(r'^\d+\.\s', line):
                content.append(Paragraph(line, self.styles['CustomBullet']))
            
            # Regular text
            else:
                # Handle bold formatting within text
                text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                content.append(Paragraph(text, self.styles['CustomBody']))
        
        return content
    
    def generate_pdf(self, study_notes: str, title: str = "Study Notes") -> bytes:
        """Generate PDF from study notes"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Add title
        content = [Paragraph(title, self.styles['CustomTitle'])]
        content.append(Spacer(1, 20))
        
        # Add study notes content
        content.extend(self.markdown_to_pdf_content(study_notes))
        
        try:
            doc.build(content)
            buffer.seek(0)
            return buffer.getvalue()
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
            return None

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from uploaded PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(uploaded_file) -> str:
    """Extract text from uploaded DOCX"""
    try:
        doc = docx.Document(uploaded_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def main():
    # Main header
    st.markdown('<h1 class="main-header">📚 Medical Study Notes Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Model selection
    model_options = [
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022", 
        "claude-3-opus-20240229",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "gemini-1.5-pro",
        "gemini-1.5-flash"
    ]
    
    selected_model = st.sidebar.selectbox(
        "Select LLM Model",
        model_options,
        index=0
    )
    
    # API Key input
    api_key = st.sidebar.text_input(
        "Enter API Key",
        type="password",
        placeholder="Enter your API key here..."
    )
    
    # Content type selection
    content_types = [
        "medical textbook chapter",
        "research paper",
        "clinical guidelines",
        "medical article",
        "case study",
        "review article"
    ]
    
    content_type = st.sidebar.selectbox(
        "Content Type",
        content_types,
        index=0
    )
    
    # Study field
    study_fields = [
        "medicine",
        "nursing",
        "pharmacy",
        "dentistry",
        "veterinary medicine",
        "public health",
        "biomedical sciences"
    ]
    
    study_field = st.sidebar.selectbox(
        "Study Field",
        study_fields,
        index=0
    )
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        chunk_size = st.slider("Chunk Size (tokens)", 1000, 5000, 3000, 500)
        overlap_size = st.slider("Overlap Size (tokens)", 100, 500, 200, 50)
    
    # Main content area
    st.markdown('<div class="sub-header">📄 Input Your Content</div>', unsafe_allow_html=True)
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload File", "Paste Text"],
        horizontal=True
    )
    
    text_content = ""
    
    if input_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload your document",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        if uploaded_file is not None:
            file_type = uploaded_file.type
            
            with st.spinner("Extracting text from file..."):
                if file_type == "application/pdf":
                    text_content = extract_text_from_pdf(uploaded_file)
                elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text_content = extract_text_from_docx(uploaded_file)
                elif file_type == "text/plain":
                    text_content = str(uploaded_file.read(), "utf-8")
            
            if text_content:
                st.markdown('<div class="success-box">✅ File uploaded and text extracted successfully!</div>', 
                          unsafe_allow_html=True)
                st.info(f"Extracted {len(text_content)} characters from the document.")
    
    else:  # Paste Text
        text_content = st.text_area(
            "Paste your content here:",
            height=300,
            placeholder="Paste your medical text, research paper, or textbook content here..."
        )
    
    # Text preview
    if text_content:
        with st.expander("📖 Preview Text (First 500 characters)"):
            st.text(text_content[:500] + "..." if len(text_content) > 500 else text_content)
    
    # Generate study notes
    if st.button("🎯 Generate Study Notes", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter your API key in the sidebar.")
            return
        
        if not text_content:
            st.error("Please provide text content to process.")
            return
        
        try:
            # Initialize processors
            text_processor = TextProcessor()
            llm_processor = LLMProcessor(selected_model, api_key)
            
            # Clean and process text
            with st.spinner("Preprocessing text..."):
                cleaned_text = text_processor.clean_text(text_content)
                chunks = text_processor.recursive_text_split(
                    cleaned_text, 
                    max_chunk_size=chunk_size, 
                    overlap=overlap_size
                )
            
            st.info(f"Text split into {len(chunks)} chunks for processing.")
            
            # Generate study notes for each chunk
            all_study_notes = []
            progress_bar = st.progress(0)
            
            for i, chunk in enumerate(chunks):
                with st.spinner(f"Generating study notes for chunk {i+1}/{len(chunks)}..."):
                    study_notes = llm_processor.generate_study_notes(
                        chunk, content_type, study_field
                    )
                    all_study_notes.append(study_notes)
                    progress_bar.progress((i + 1) / len(chunks))
                
                # Add delay to avoid rate limiting
                if i < len(chunks) - 1:
                    time.sleep(1)
            
            # Combine all study notes
            combined_notes = "\n\n---\n\n".join(all_study_notes)
            
            # Display results
            st.markdown('<div class="success-box">✅ Study notes generated successfully!</div>', 
                      unsafe_allow_html=True)
            
            # Study notes display
            st.markdown("## 📝 Generated Study Notes")
            st.markdown(combined_notes)
            
            # Store in session state for download
            st.session_state.study_notes = combined_notes
            st.session_state.notes_title = f"{content_type.title()} - Study Notes"
            
            # Download section
            st.markdown("## 📥 Download Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Text download
                st.download_button(
                    label="📄 Download as Text",
                    data=combined_notes,
                    file_name=f"study_notes_{int(time.time())}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                # PDF download
                if st.button("📊 Generate PDF", use_container_width=True):
                    with st.spinner("Generating PDF..."):
                        pdf_generator = PDFGenerator()
                        pdf_bytes = pdf_generator.generate_pdf(
                            combined_notes, 
                            st.session_state.notes_title
                        )
                        
                        if pdf_bytes:
                            st.download_button(
                                label="📑 Download PDF",
                                data=pdf_bytes,
                                file_name=f"study_notes_{int(time.time())}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Instructions and information
    with st.expander("📋 Instructions & Information"):
        st.markdown("""
        ### How to Use:
        1. **Configure Settings**: Select your preferred LLM model and enter the API key in the sidebar
        2. **Choose Content Type**: Select the type of medical content you're processing
        3. **Input Content**: Either upload a file (PDF, DOCX, TXT) or paste text directly
        4. **Generate Notes**: Click the "Generate Study Notes" button to process your content
        5. **Download**: Get your formatted study notes as text or PDF
        
        ### Supported LLM Models:
        - **Claude Models**: claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus
        - **GPT Models**: gpt-4-turbo, gpt-4, gpt-3.5-turbo  
        - **Gemini Models**: gemini-1.5-pro, gemini-1.5-flash
        
        ### Features:
        - ✅ Comprehensive text preprocessing and chunking
        - ✅ Multiple LLM integration support
        - ✅ Medical content specialization
        - ✅ Structured note formatting
        - ✅ PDF and text export options
        - ✅ Preserves all important medical information
        
        ### Note Format Features:
        - **Complete Content Preservation**: No summarization or omission
        - **Structured Bullet Format**: Clear hierarchy and organization
        - **Medical Specialization**: Preserves drug names, dosages, clinical findings
        - **Academic Level**: Tailored for postgraduate students
        - **Professional Formatting**: Bold headings, proper indentation, numbering
        """)

if __name__ == "__main__":
    main()