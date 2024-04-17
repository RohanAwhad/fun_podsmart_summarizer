# Fun PodSmart Summarizer

The Fun PodSmart Summarizer is a Python-based application designed to automatically generate concise summaries of textual content. Utilizing advanced language processing techniques, this tool is ideal for digesting large volumes of text quickly and efficiently.

## Features

- **API for Text Summarization**: Offers a FastAPI interface to summarize text through HTTP requests.
- **Advanced NLP Operations**: Employs several stages of summarization and topic extraction, leveraging the LangChain library and custom utilities.
- **Visualization Tools**: Includes functionality to visualize similarity matrices and topic distributions.
- **PDF Text Extraction**: Capable of reading and cleaning text from PDF files.
- **JSONL to Markdown Conversion**: Converts summaries from JSONL format to markdown files.

## Installation

To set up the project, follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/rohanawhad/fun_podsmart_summarizer.git
   ```

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   - `API_ACCESS_KEY`: Your custom access key for using the API.

## Usage

### Starting the Server

Run the following command to start the FastAPI server:

```
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Summarizing Text

Send a POST request to the `/summarize` endpoint with the following JSON body:

```json
{
  "text": "Your text here",
  "access_key": "Your API access key"
}
```

The API will return a markdown-formatted summary of the provided text.

### Visualizing Data

Utilize the provided plotting functions in the `main.py` file to generate plots for similarity matrices and topic distributions.

## Development

### Modules

- **API Module (`api.py`)**: Handles the web server and API requests.
- **Main Logic (`main.py`)**: Contains the core functionality for text summarization and visualization.
- **Utilities (`src/utils.py`)**: Provides helper functions for text processing and data manipulation.
- **Encoder Service (`src/encoder_service.py`)**: Manages sentence embedding operations.

### Adding New Features

1. Create a new branch for your feature.
2. Implement your feature with corresponding unit tests.
3. Submit a pull request to the main branch.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.
