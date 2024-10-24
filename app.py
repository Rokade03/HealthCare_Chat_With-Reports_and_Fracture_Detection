import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO


# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name = "NeuML/pubmedbert-base-embeddings")
    # GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """ After getting the PDF you will identify if the pdf is related to medical field if not then give the prompt "The PDF is not a medical document" and don't give any reply, else you are a professional medical agent who will analyze the report line by line and answer the question provied by the user in a detailed manner from the provided report as well as give remedies if needed, Make sure to provide all the details, remedies and readings, if the answer or information is not available in the context provided just return the statement, "Answer is not available in the context", Make sure you do not provide the wrong or incorrect answers to the user
     Context:\n {context}?\n
     Question:\n {question}\n 

     Answer:
    """

    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature = 0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type = "stuff", prompt = prompt)
    return chain

def user_input(user_question):
    # Load embeddings and vector store
    embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Perform a similarity search on the user question
    docs = new_db.similarity_search(user_question)

    # Load the conversational chain
    chain = get_conversational_chain()

    # Generate the response by passing input_documents (which are the results of the similarity search)
    response = chain(
        {"input_documents": docs, "question": user_question},  # Pass input_documents and question
        return_only_outputs=True
    )

    # Check if the PDF is a medical document
    if response == "The PDF is not a medical document":
        st.write("Reply: Please Provide A Medical Document")
        return None
    else:
        st.write("Reply:", response['output_text'])
        return response
    
def yolo_image_inference(image):
    model = YOLO('C:/Users/Aayush/Desktop/AIML_Project/RAG_and_YOLO_App/models/best.pt')  

    image_np = np.array(image)

    results = model.predict(source=image_np, conf=0.1)

    annotated_frame = results[0].plot()
    return Image.fromarray(annotated_frame)



def main():
    st.set_page_config("Chat with PDF & YOLO")
    st.header("Chat with Reports and Fracture Detection💁")
    st.text("Note: Please upload a report in PDF format or images for fracture detection.")

    tab1, tab2 = st.tabs(["PDF Q&A", "YOLO Object Detection"])

    with tab1:
        user_question = st.text_input("Ask a Question from the reports")

        if user_question:
            user_input(user_question)

        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Reports", accept_multiple_files=True, type=['pdf'])
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done processing the PDF!")
    with tab2:
        st.subheader("Fracture Detection")

        uploaded_image = st.file_uploader("Upload an image for object detection", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Run Detection"):
                with st.spinner("Running Detection..."):
                    detected_image = yolo_image_inference(image)
                    st.image(detected_image, caption="Detected Objects", use_column_width=True)
                    st.success("Detection complete!")



if __name__ == "__main__":
    main()