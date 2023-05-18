# chatting-with-papers
Chatting with scientific papers (in PDF format).


# Setup

## Environment

1. Create a virtual environment.
2. Within the environment, install the dependencies with `pip install -r requirements.txt`.

## API Key

Copy your OpenAI API key in a file `api.key`.

# Run the Script

Run the arxivbot from the command line. For help: `python arxivbot.py --help`.

Example:
```
python arxivbot.py --ask "What is the cure time of DA 409 as recommended by the manufacturer?" -i polymers\

```

If you don't have access to the GPT-4 API yet, you need to switch the type of model you're using. 

# Data
The test data are freely available scientific papers about polymers. You can [download them here] (https://drive.google.com/drive/folders/1NrDX9KQmqnbqrg7yx-Y5FgWelC24TnGQ?usp=sharing)

However, our small bot will work with any corpus of documents in PDF format. Feel free to experiment with it.
