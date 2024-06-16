import streamlit as st
import app  as app  
from pathlib import Path
import os
import time

st.set_page_config(
    page_title="AWS Bedrock RAG Demo",
    page_icon="🧠",
)

st.write("## AWS Bedrock Generative-AI (RAG)✍️")


with st.form("my-form", clear_on_submit=True):
    prompt =st.text_input("Enter your prompt",)
    btngenerate=st.form_submit_button("Generate")
    if btngenerate:
        vec_str=app.vector_retriever(prompt)
        # st.write(vec_str)
        llm=app.get_llm()
        resp=app.get_response_from_llm(llm,vec_str,app.prompt_template,prompt)
        st.markdown("👧: " + prompt)
        st.markdown("🤖: " + resp)


st.sidebar.write("# Upload Source PDF Here!")
with st.sidebar.form("mysideform", clear_on_submit=True):
    File = st.file_uploader(label = "Select only PDF File", type=["pdf"])
    btnupload=st.form_submit_button("Upload")
    if btnupload:
        if File:
            save_path = Path('D:\RAG Projects\AWS Bedrock - RAG\source_data', File.name)
            with open(save_path, mode='wb') as w:
                w.write(File.getvalue())

            if save_path.exists():
                alert=st.sidebar.success(f'File {File.name} is successfully saved!')
                time.sleep(3) # Wait for 3 seconds
                alert.empty() # Clear the alert
                # st.write(save_path)
                st.markdown('''
                <style>
                    .stFileUploaderFile {display: none}
                <style>''',
                unsafe_allow_html=True)
                File = []
                # Re-Generate Vector
                with st.spinner('Data Ingestion in progress...'):              
                    doc=app.data_ingestion() 
                    app.get_vector_store(doc)
                sts_success=st.success('Data Ingestion Completed!')
                time.sleep(3) # Wait for 3 seconds
                sts_success.empty() # Clear the alert







