from dotenv import load_dotenv
load_dotenv()
import base64
import streamlit as st
import os     
import io
from PIL import Image 
import pdf2image
import google.generativeai as genai
import pandas as pd
from pathlib import Path
import tempfile
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re  
import time   


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))    

if 'results' not in st.session_state:
    st.session_state.results = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

def get_gemini_response(input, pdf_content, prompt):
    """Get response from Gemini model"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input, pdf_content[0], prompt])    
    return response.text   

def input_pdf_setup(uploaded_file):
    """Process a PDF file and convert first page to image for Gemini"""
    if uploaded_file is not None:
        # Convert the PDF to image
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file Found or Uploaded 🔍")

def parse_percentage(text):
    """Extract percentage match from text"""
    percentage_patterns = [
        r"(\d{1,3})%\s+Match",
        r"Match\s+Percentage:\s*(\d{1,3})%",
        r"Percentage\s+Match:\s*(\d{1,3})%",
        r"(\d{1,3})%\s+match",
        r"match\s+percentage\s+is\s+(\d{1,3})%",
        r"(\d{1,3})%"
    ]
    
    for pattern in percentage_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                percentage = int(match.group(1))
                return min(percentage, 100)  # Cap at 100%
            except ValueError:
                continue
    
    return 0  # Default if no percentage found

def extract_candidate_name(text):
    """Extract candidate name from text"""
    # Look for patterns like "Name: John Doe" or "Candidate: John Doe"
    name_patterns = [
        r"Name:\s*([\w\s]+)",
        r"Candidate(?:'s)?\s*Name:\s*([\w\s]+)",
        r"Candidate:\s*([\w\s]+)"
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no match, look for something that looks like a name at the beginning
    lines = text.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        if len(line.strip()) > 0 and len(line.strip().split()) <= 4:
            return line.strip()
    
    return "Unknown Candidate"

def extract_keywords(text, section_name):
    """Extract keywords from a specific section"""
    # Define patterns to look for
    section_patterns = [
        f"{section_name}[:\s]+(.*?)(?:\n\n|\n[A-Z])",
        f"{section_name}[:\s]+((?:\n\s*-\s*.*)+)"
    ]
    
    keywords = []
    
    # Try finding the section with the patterns
    for pattern in section_patterns:
        matches = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            content = matches.group(1).strip()
            
            # Check if content has bullet points
            if '-' in content or '•' in content:
                # Split by bullet points
                for item in re.split(r'\n\s*[-•]\s*', content):
                    if item.strip():
                        keywords.append(item.strip())
            else:
                # Split by commas or newlines
                for item in re.split(r'[,\n]', content):
                    if item.strip():
                        keywords.append(item.strip())
            
            break
    
    # Clean up the keywords
    cleaned_keywords = []
    for keyword in keywords:
        keyword = keyword.strip().rstrip('.,:;')
        if keyword and len(keyword) > 1:
            cleaned_keywords.append(keyword)
    
    return cleaned_keywords[:10]  # Limit to 10 keywords for consistency

def extract_structured_data(raw_text, filename):
    """Extract structured data from the Gemini API text response"""
    
    # First try to parse as JSON
    try:
        data = json.loads(raw_text)
        # If successful, ensure all required fields exist
        if 'candidate_name' not in data:
            data['candidate_name'] = extract_candidate_name(raw_text)
        return data
    except json.JSONDecodeError:
        # JSON parsing failed, extract information manually
        pass
    
    # Extract data manually
    structured_data = {}
    
    # Basic metadata
    structured_data['filename'] = filename
    structured_data['candidate_name'] = extract_candidate_name(raw_text)
    structured_data['percentage_match'] = parse_percentage(raw_text)
    
    # Extract keywords
    structured_data['matching_keywords'] = extract_keywords(raw_text, "Matching Keywords")
    if not structured_data['matching_keywords']:
        structured_data['matching_keywords'] = extract_keywords(raw_text, "Skills Found")
    
    structured_data['missing_keywords'] = extract_keywords(raw_text, "Missing Keywords")
    if not structured_data['missing_keywords']:
        structured_data['missing_keywords'] = extract_keywords(raw_text, "Skills Missing")
    
    # Extract strengths and areas for improvement
    structured_data['strengths'] = extract_keywords(raw_text, "Strengths")
    structured_data['areas_for_improvement'] = extract_keywords(raw_text, "Areas for Improvement")
    if not structured_data['areas_for_improvement']:
        structured_data['areas_for_improvement'] = extract_keywords(raw_text, "Areas of Improvement")
    
    # Extract recommendation
    recommendation_pattern = r"(?:Conclusion|Summary|Recommendation)[:\s]+(.*?)(?:\n\n|\Z)"
    recommendation_match = re.search(recommendation_pattern, raw_text, re.DOTALL | re.IGNORECASE)
    if recommendation_match:
        structured_data['recommendation'] = recommendation_match.group(1).strip()
    else:
        structured_data['recommendation'] = "No recommendation provided."
    
    # Add placeholder data
    structured_data['experience_years'] = estimate_experience(raw_text)
    structured_data['education_level'] = estimate_education(raw_text)
    structured_data['skills'] = generate_skills_data(structured_data['matching_keywords'])
    
    return structured_data

def estimate_experience(text):
    """Estimate years of experience from text"""
    experience_patterns = [
        r"(\d+)\s+years?\s+of\s+experience",
        r"experience\s*:?\s*(\d+)\s+years",
        r"(\d+)\s+years?\s+in\s+the\s+industry"
    ]
    
    for pattern in experience_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                years = int(match.group(1))
                return min(years, 30)  # Cap at 30 years
            except ValueError:
                continue
    
    # If no explicit mention, estimate based on other factors
    if "senior" in text.lower() or "lead" in text.lower() or "manager" in text.lower():
        return 5
    elif "junior" in text.lower() or "entry" in text.lower():
        return 1
    else:
        return 3  # Default assumption

def estimate_education(text):
    """Estimate education level from text"""
    if re.search(r"ph\.?d\.?|doctor(?:ate)?", text, re.IGNORECASE):
        return "PhD"
    elif re.search(r"master(?:'s)?|ms|m\.s\.|mba|m\.b\.a\.", text, re.IGNORECASE):
        return "Master's"
    elif re.search(r"bachelor(?:'s)?|bs|b\.s\.|ba|b\.a\.", text, re.IGNORECASE):
        return "Bachelor's"
    else:
        return "Bachelor's"  # Default assumption

def generate_skills_data(keywords):
    """Generate skills data from keywords"""
    skills = []
    
    for i, keyword in enumerate(keywords[:8]):  # Limit to 8 skills
        # Generate a skill level between 6-10 for matching skills
        level = min(10, 6 + i % 5)
        skills.append({
            "name": keyword,
            "level": level
        })
    
    return skills

def process_multiple_resumes(uploaded_files, job_description):
    """Process multiple resumes and return analysis results"""
    results = []
    progress_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        try:
            # Update progress
            progress_bar.progress((i) / len(uploaded_files))
            
            # Get PDF content for Gemini
            pdf_content = input_pdf_setup(file)
            
            # Use the percentage match prompt for consistent responses
            match_response = get_gemini_response(multiple_resume_prompt, pdf_content, job_description)
            
            # Extract structured data from the text response
            resume_data = extract_structured_data(match_response, file.name)
            
            # Add to results if we got valid data
            if resume_data and resume_data.get('percentage_match', 0) > 0:
                results.append(resume_data)
            else:
                st.warning(f"Could not extract match data for {file.name}")
                
            # Small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    
    # Complete the progress bar
    progress_bar.progress(1.0)
    
    # Sort results by percentage match (descending)
    if results:
        results.sort(key=lambda x: x.get('percentage_match', 0), reverse=True)
    
    return results

def save_to_csv(results, filename="resume_analysis.csv"):
    """Save analysis results to CSV file"""
    # Flatten the skills list for CSV
    flat_results = []
    for result in results:
        flat_result = result.copy()
        flat_result['skills'] = ', '.join([skill['name'] for skill in result.get('skills', [])])
        flat_result['matching_keywords'] = ', '.join(result.get('matching_keywords', []))
        flat_result['missing_keywords'] = ', '.join(result.get('missing_keywords', []))
        flat_result['strengths'] = ', '.join(result.get('strengths', []))
        flat_result['areas_for_improvement'] = ', '.join(result.get('areas_for_improvement', []))
        flat_results.append(flat_result)
        
    df = pd.DataFrame(flat_results)
    csv_data = df.to_csv(index=False)
    return csv_data

def save_to_excel(results, filename="resume_analysis.xlsx"):
    """Save analysis results to Excel file"""
    # Flatten the skills list for Excel
    flat_results = []
    for result in results:
        flat_result = result.copy()
        flat_result['skills'] = ', '.join([skill['name'] for skill in result.get('skills', [])])
        flat_result['matching_keywords'] = ', '.join(result.get('matching_keywords', []))
        flat_result['missing_keywords'] = ', '.join(result.get('missing_keywords', []))
        flat_result['strengths'] = ', '.join(result.get('strengths', []))
        flat_result['areas_for_improvement'] = ', '.join(result.get('areas_for_improvement', []))
        flat_results.append(flat_result)
        
    df = pd.DataFrame(flat_results)
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_data = excel_buffer.getvalue()
    return excel_data

def save_to_json(results, filename="resume_analysis.json"):
    """Save analysis results to JSON file"""
    return json.dumps(results, indent=4)

def create_comparison_chart(results):
    """Create a comparison chart of candidates"""
    if not results:
        return None
        
    df = pd.DataFrame(results)
    df = df[['candidate_name', 'percentage_match']]
    
    # Extract the top 10 candidates for better visualization
    if len(df) > 10:
        df = df.head(10)
    
    # Create horizontal bar chart for percentage match
    fig = px.bar(
        df, 
        y='candidate_name', 
        x='percentage_match',
        text='percentage_match',
        labels={'percentage_match': 'Match %', 'candidate_name': 'Candidate'},
        title='Candidate Match Comparison',
        orientation='h',
        color='percentage_match',
        color_continuous_scale='viridis',
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_range=[0, 100]
    )
    
    return fig

def create_skills_comparison(results, top_n=5):
    """Create a comparison of skills across top candidates"""
    if not results or len(results) < 2:
        return None
        
    # Extract top N candidates
    top_candidates = results[:top_n]
    
    # Create skills data
    skills_data = []
    for candidate in top_candidates:
        for skill in candidate.get('skills', []):
            skills_data.append({
                'candidate': candidate['candidate_name'],
                'skill': skill['name'],
                'level': skill['level'],
            })
    
    if not skills_data:
        return None
        
    # Create DataFrame
    skills_df = pd.DataFrame(skills_data)
    
    # Create heatmap
    fig = px.density_heatmap(
        skills_df, 
        x='skill', 
        y='candidate',
        z='level',
        title='Skills Comparison',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis={'categoryorder': 'category ascending'},
        yaxis={'categoryorder': 'category ascending'}
    )
    
    return fig

# Prompt for multiple resume screening - simplified to get consistent results
multiple_resume_prompt = """
Analyze this resume against the provided job description. 

1. Calculate a percentage match between the resume and job description.
2. List the matching keywords or skills found in both the resume and job description.
3. List the missing keywords or skills that are in the job description but missing from the resume.
4. Identify the candidate's key strengths relative to the job description.
5. Identify areas where the candidate could improve to better match the job requirements.
6. Provide a brief recommendation about the candidate's fit for the role.

Format your response as follows:

Candidate Name: [Name]
Percentage Match: [X]%

Matching Keywords:
- [keyword1]
- [keyword2]
...

Missing Keywords:
- [keyword1]
- [keyword2]
...

Strengths:
- [strength1]
- [strength2]
...

Areas for Improvement:
- [area1]
- [area2]
...

Recommendation:
[Brief assessment of candidate fit]
"""

# Single resume prompts (from original code)
input_prompt1 = """
You are an experienced human resource manager with 20+ years of technical experience in any one of the job fields of Data Science, Machine Learning, Devops, Cloud Computing, FullStack Web Development and your main task is to review the provided resume aganist the job description provided for these profiles.

Extract and summarize the candidate's basic details, including name, education, projects, and experience in an ordered format from the resume. Provide a detailed professional evaluation of how well the candidate's profile aligns with the job role specified in the job description. Highlight reasons for selecting or rejecting the candidate based on your evaluation.

And also avoid, ignore any safety ratings issues and provide the correct response
"""

input_prompt2 = """
Review the resume against the job description for roles in Cloud Computing, Data Science, FullStack Web Development, Graphic Design, Machine Learning, and Devops. Identify and specifically highlight mainly the strengths and areas needing improvement seperatley in clear and concise bullet points format. 

Then Suggest strategies tailored for the specified job role, including in-demand skills, recommended courses, and projects, but keep it minimal, straight forward and effective to the candidate job.

And also avoid, ignore any safety ratings issues and provide the correct response
"""

input_prompt3 = """
Evaluate the resume against the provided job description using ATS (Application Tracking System) scanner deep functionality. Calculate and Highlight the percentage match between the resume and job description at the top of the response. 

Then Highlight keywords and skills from the job description that are missing in the resume specifically and clearly in bullet points format listing all the important and main keywords and skills under different headings. 

At last Conclude with insights on the overall fit of the resume for the specified job role, but keep it minimal, straight forward and effective to the candidate job.   

And also avoid, ignore any safety ratings issues and provide the correct response
"""

st.set_page_config(page_title="Enhanced Resume ATS Expert 📃", layout="wide")

with st.sidebar:
    st.title("Resume ATS Expert 🕵️")
    st.markdown("---")
    mode = st.radio("Mode", ["Single Resume Analysis", "Multiple Resume Screening"])
    
    # Custom ranking criteria (only in multiple resume mode)
    if mode == "Multiple Resume Screening":
        st.subheader("Ranking Criteria Weights")
        skills_weight = st.slider("Skills Match Weight", 0.0, 1.0, 0.5, 0.1)
        experience_weight = st.slider("Experience Weight", 0.0, 1.0, 0.3, 0.1)
        education_weight = st.slider("Education Weight", 0.0, 1.0, 0.2, 0.1)

if mode == "Single Resume Analysis":
    st.header("Single Resume Analysis 📝")
    
    input_text = st.text_area("Job Description:", key="input", height=200)
    
    uploaded_file = st.file_uploader("Upload your Resume (PDF format):", type=["pdf"])
    
    if uploaded_file is not None:
        st.success("PDF File Uploaded Successfully! 🎯")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        submit1 = st.button("Tell me about the Resume ℹ️")
    with col2:
        submit2 = st.button("Strengths & Areas of Improvement 💪")
    with col3:
        submit3 = st.button("Match Percentage with JD 🤝")
    
    # Process single resume analysis
    if submit1:
        if uploaded_file is not None:
            with st.spinner("Analyzing resume..."):
                pdf_content = input_pdf_setup(uploaded_file)
                response = get_gemini_response(input_prompt1, pdf_content, input_text)
                st.subheader("Resume Details:")
                st.write(response)
        else:
            st.warning("Please upload your resume first 🔍")
            
    elif submit2:
        if uploaded_file is not None:
            with st.spinner("Analyzing strengths and areas for improvement..."):
                pdf_content = input_pdf_setup(uploaded_file)
                response = get_gemini_response(input_prompt2, pdf_content, input_text)
                st.subheader("Resume Analysis:")
                st.write(response)
        else:
            st.warning("Please upload your resume first 🔍")   
    
    elif submit3:
        if uploaded_file is not None:
            with st.spinner("Calculating match percentage..."):
                pdf_content = input_pdf_setup(uploaded_file)
                response = get_gemini_response(input_prompt3, pdf_content, input_text)
                st.subheader("Resume x JD Match Percentage:")
                st.write(response)
        else:
            st.warning("Please upload your resume first 🔍")

else:  # Multiple Resume Screening
    st.header("Multiple Resume Screening & Ranking 📊")
    
    input_text = st.text_area("Job Description:", key="input_multi", height=200)
    
    uploaded_files = st.file_uploader("Upload Resumes (PDF format):", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} Resumes Uploaded Successfully! 🎯")
    
    # Process button - this only runs if we haven't already processed
    process_button = st.button("Screen and Rank Resumes 🔍")
    
    if process_button and not st.session_state.processed:
        if uploaded_files and input_text:
            with st.spinner(f"Processing {len(uploaded_files)} resumes... This might take a while."):
                # Process all resumes
                results = process_multiple_resumes(uploaded_files, input_text)
                
                if results:
                    # Apply custom ranking if weights are changed
                    if 'skills_weight' in locals():
                        for result in results:
                            custom_score = (
                                result.get('percentage_match', 0) * skills_weight +
                                min(result.get('experience_years', 0) * 10, 100) * experience_weight +
                                ({"Bachelor's": 60, "Master's": 80, "PhD": 100}.get(result.get('education_level', ""), 50)) * education_weight
                            )
                            result['custom_score'] = round(custom_score, 2)
                        
                        # Sort by custom score if available
                        results.sort(key=lambda x: x.get('custom_score', 0), reverse=True)
                    
                    # Store in session state
                    st.session_state.results = results
                    st.session_state.processed = True
                    
                    # Ensure we rerun to display results
                    st.rerun()
                else:
                    st.error("No results were generated. There might be an issue with processing the resumes.")
            
        elif not uploaded_files:
            st.warning("Please upload at least one resume 🔍")
        else:
            st.warning("Please provide a job description 📝")

    # Display results if they exist in session state
    if st.session_state.processed and st.session_state.results:
        results = st.session_state.results
        
        # Display results
        st.subheader("Resume Screening Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Ranked List", "Detailed View", "Comparison Charts", "Export"])
        
        with tab1:
            # Display ranked list of candidates
            for i, result in enumerate(results):
                with st.expander(f"#{i+1}: {result['candidate_name']} - Match: {result['percentage_match']}%"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Matching Keywords:**")
                        st.write(", ".join(result['matching_keywords']))
                        
                        st.write("**Strengths:**")
                        for strength in result['strengths']:
                            st.write(f"✓ {strength}")
                            
                    with col2:
                        st.write("**Missing Keywords:**")
                        st.write(", ".join(result['missing_keywords']))
                        
                        st.write("**Areas for Improvement:**")
                        for area in result['areas_for_improvement']:
                            st.write(f"- {area}")
                    
                    st.write("**Recommendation:**")
                    st.write(result['recommendation'])
        
        with tab2:
            # Detailed comparison view
            st.subheader("Detailed Candidate Information")
            
            # Use a key for the multiselect to prevent state issues
            if 'selected_candidates' not in st.session_state:
                st.session_state.selected_candidates = [results[0]['candidate_name']] if results else []
                
            selected_candidates = st.multiselect(
                "Select candidates to compare:",
                options=[result['candidate_name'] for result in results],
                default=st.session_state.selected_candidates,
                key='candidate_multiselect'
            )
            
            # Update the session state
            st.session_state.selected_candidates = selected_candidates
            
            if selected_candidates:
                # Create detailed comparison table
                comparison_data = []
                for result in results:
                    if result['candidate_name'] in selected_candidates:
                        comparison_data.append(result)
                
                # Convert to DataFrame for better display
                df = pd.DataFrame(comparison_data)
                
                # Select columns to display
                display_cols = ['candidate_name', 'percentage_match', 'experience_years', 'education_level']
                if 'custom_score' in df.columns:
                    display_cols.insert(2, 'custom_score')
                    
                st.dataframe(df[display_cols])
                
                # Show individual candidate details
                for candidate in comparison_data:
                    with st.expander(f"Detailed Profile: {candidate['candidate_name']}"):
                        # Skills radar chart
                        if candidate['skills']:
                            skills_df = pd.DataFrame(candidate['skills'])
                            fig = px.line_polar(
                                skills_df, r='level', theta='name', line_close=True,
                                title=f"{candidate['candidate_name']}'s Skills Profile"
                            )
                            st.plotly_chart(fig)
                        
                        # Other details
                        st.write("**Education Level:**", candidate['education_level'])
                        st.write("**Years of Experience:**", candidate['experience_years'])
                        st.write("**Matching Keywords:**", ", ".join(candidate['matching_keywords']))
                        st.write("**Missing Keywords:**", ", ".join(candidate['missing_keywords']))
                        st.write("**Recommendation:**", candidate['recommendation'])
        
        with tab3:
            # Comparison charts
            st.subheader("Candidate Comparison Charts")
            
            # Overall match chart
            match_chart = create_comparison_chart(results)
            if match_chart:
                st.plotly_chart(match_chart, use_container_width=True)
            
            # Skills comparison
            top_n = min(5, len(results))
            if top_n >= 2:
                skills_chart = create_skills_comparison(results, top_n)
                if skills_chart:
                    st.plotly_chart(skills_chart, use_container_width=True)
        
        with tab4:
            # Export options
            st.subheader("Export Results")
            export_format = st.selectbox(
                "Select export format:",
                options=["CSV", "Excel", "JSON"]
            )
            
            if export_format == "CSV":
                csv_data = save_to_csv(results)
                date_str = datetime.now().strftime("%Y%m%d")
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"resume_analysis_{date_str}.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                excel_data = save_to_excel(results)
                date_str = datetime.now().strftime("%Y%m%d")
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name=f"resume_analysis_{date_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:  # JSON
                json_data = save_to_json(results)
                date_str = datetime.now().strftime("%Y%m%d")
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"resume_analysis_{date_str}.json",
                    mime="application/json"
                )

    # Reset button to clear results and process again
    if st.session_state.processed:
        if st.button("Reset and Process New Resumes"): 
            st.session_state.processed = False
            st.session_state.results = None
            st.rerun()

# Footer
st.markdown("---")    
st.markdown("Resume ATS Expert | Enhanced Version with Multiple Resume Screening")   