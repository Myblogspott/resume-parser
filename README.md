# ATS Friendly Resume Parser(Beta) by Sai Raghavendra Maddula



This project uses a multi-agent architecture (with RAG, vector databases, and a web scrapping component) to generate ATS-friendly resumes and handle job application processes. This README describes the initial setup and how to run the main scripts.

## Setup Instructions

### 1. Create a Virtual Environment and Activate It

Create a new virtual environment for the project and activate it:

- **On macOS/Linux:**

  ```bash
  python -m venv venv
  source venv/bin/activate
  ```

- **On Windows:**

  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

### 2. Install Dependencies

Install the required Python packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Run the Application Scripts

After the dependencies are installed, run the main scripts:

- **Run the RAG Pipeline Script:**

  ```bash
  python RAG.py
  ```

- **Run the Web Scrapping Script:**

  ```bash
  python webscrapping.py
  ```

## Project Structure

- **RAG.py:**  
  Contains the retrieval-augmented generation pipeline that uses vector embeddings and a text-generation model to synthesize a new ATS-friendly resume.

- **webscrapping.py:**  
  Contains the web scrapping agent that navigates job application steps and interacts with job portals.

- **requirements.txt:**  
  Lists all the dependencies needed for the project.

## Additional Information

- Ensure you have the necessary GPU setup if you plan to use GPU-enabled models.
- If you encounter issues with FAISS, make sure you have installed either `faiss-cpu` or `faiss-gpu` as appropriate.
- Modify file paths in the scripts as needed to match your environment.
