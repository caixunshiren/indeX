# indeX
indeX your multimodal training data



### Setup

```bash
./setup.sh
```

Create a .env file in the root directory with the following variables (see .env_sample):

```bash
API_PROVIDER=OPENAI # OPENAI or XAI
XAI_API_KEY=YOUR_XAI_KEY
OPENAI_API_KEY=YOUR_OPENAI_KEY
```

### Run

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

```bash
python app.py image-test
```