# RAG From Scratch: Chatting With Scientific Papers
Chatting with scientific papers (in PDF format).


# Setup

## Environment

1. Create a virtual environment.
2. Within the environment, install the dependencies with `pip install -r requirements.txt`.

The requirements are managed with `pip-compile` you can regenerate the `requirements.txt` from `requirements.in`.

## API Key

1. Get an OpenAI API key.
2. Copy your OpenAI API key in a .env file like this:

```
OPENAI_API_KEY="<YOUR-KEY>"
```

# Run the Script

Run the arxivbot from the command line. For help: `python arxivbot.py --help`.


# Data
The test data are freely available scientific papers about polymers.

However, our small bot will work with any corpus of documents in PDF format. Feel free to experiment with it.
