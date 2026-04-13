# gRPC LLM eemplate

A production-ready template for serving Large Language Models via gRPC with streaming token generation. Built with Python, PyTorch, Hugging Face Transformers, and gRPC.

## Features

- gRPC-based API for LLM inference
- Streaming token generation for real-time responses
- Configurable model selection via CLI or config file
- Threaded gRPC server for concurrent requests
- Support for any HuggingFace causal language model

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster inference)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ashankgupta/grpc_llm_template.git
cd grpc-llm-template
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Generate gRPC stubs:

```bash
python -m grpc_tools.protoc -I./proto --python_out=./generated --grpc_python_out=./generated proto/llm.proto
```

## Configuration

Edit `config.yaml` to customize server and model settings:

```yaml
server:
  host: "0.0.0.0"
  port: 50051

model:
  name: "gpt2"
  max_tokens: 50
```

| Parameter | Description |
|-----------|-------------|
| `server.host` | Network interface to bind |
| `server.port` | gRPC server port |
| `model.name` | HuggingFace model identifier |
| `model.max_tokens` | Maximum tokens to generate |

## Usage

### Starting the Server

```bash
python server/server.py
```

Override configuration via command-line arguments:

```bash
python server/server.py --model gpt2 --port 50051
```

### Running the Client

```bash
python client/client.py "Your prompt here"
```

### Using with Another gRPC Client

The service exposes the following interface:

```protobuf
service LLMService {
  rpc Generate (Prompt) returns (stream Token);
}

message Prompt {
  string text = 1;
}

message Token {
  string text = 1;
}
```

Example Python client usage:

```python
import grpc
from generated import llm_pb2, llm_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = llm_pb2_grpc.LLMServiceStub(channel)

response = stub.Generate(llm_pb2.Prompt(text="Tell me a joke"))

for token in response:
    print(token.text, end="", flush=True)
print()
```

## Project Structure

```
grpc-llm-template/
├── client/
│   ├── __init__.py
│   └── client.py           # gRPC client implementation
├── generated/
│   ├── __init__.py
│   ├── llm_pb2.py          # Generated protobuf code
│   └── llm_pb2_grpc.py     # Generated gRPC stubs
├── proto/
│   └── llm.proto           # Protocol buffer definition
├── server/
│   ├── __init__.py
│   ├── config.py           # Configuration loader
│   ├── generator.py       # Token streaming logic
│   ├── model_loader.py     # Model loading utility
│   └── server.py           # gRPC server
├── config.yaml             # Configuration file
└── requirements.txt       # Python dependencies
```

## Supported Models

Any causal language model from Hugging Face Model Hub is supported. Examples:

- `gpt2`
- `gpt2-medium`
- `gpt2-large`
- `EleutherAI/gpt-neo-125M`
- `meta-llama/Llama-2-7b-hf`

## Performance Considerations

- Use GPU acceleration by ensuring PyTorch is installed with CUDA support
- Adjust `max_workers` in `server.py` for concurrent request handling
- Consider model quantization for memory-constrained environments

## License

MIT License
