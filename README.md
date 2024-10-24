
# RAG and YOLO Application

This project integrates **RAG (Retrieval-Augmented Generation)** for question-answering and **YOLO (You Only Look Once)** for object detection. The application processes input documents (PDFs) and images to perform efficient document-based Q&A and real-time object detection.

## Features

- **PDF Q&A**: Extracts information from PDF files and answers questions based on the content.
- **YOLO Object Detection**: Detects objects in real-time or uploaded images using YOLOv5.
- **Efficient Search**: Utilizes RAG for advanced document retrieval and generates human-like answers.
- **Multi-language Support**: Supports a wide variety of document and image formats.
  
## Installation

To run the application locally, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the application

To start the web application, run:

```bash
python app.py
```

The application will run locally at `http://localhost:5000/`.

## Usage

### PDF Q&A

1. Upload a PDF document.
2. Ask any question related to the content of the document.
3. The system will retrieve relevant information and generate a human-like answer.

### YOLO Object Detection

1. Upload an image or use a connected webcam.
2. The YOLO model will detect objects in real time and display the results.

## File Size Restrictions

**Note**: The application uses libraries like `torch` for deep learning, which may include large files. Ensure that your repository does not exceed GitHub's file size limit (100 MB). Add large files such as `torch_cpu.dll` and `dnnl.lib` to `.gitignore`:

```plaintext
RAG_and_YOLO_App/Lib/site-packages/torch/lib/torch_cpu.dll
RAG_and_YOLO_App/Lib/site-packages/torch/lib/dnnl.lib
```

## Technologies Used

- **Python**: Backend logic and handling.
- **RAG (Retrieval-Augmented Generation)**: Document retrieval and question-answering.
- **YOLOv5**: Object detection in images.
- **Flask**: Web framework to serve the application.
- **PyTorch**: Deep learning framework for model training and inference.

## Contribution

If you want to contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Make your changes.
4. Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
