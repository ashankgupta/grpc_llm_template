import grpc
import sys

sys.path.append("..")
from concurrent import futures
import uuid

from generated import llm_pb2
from generated import llm_pb2_grpc

from server.model_loader import load_model
from server.generator import stream_tokens
from server.config import load_config

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model", help="HF model name")
parser.add_argument("--port", type=int, help="Server port")
parser.add_argument("--temperature", type=float, help="Default temperature")
parser.add_argument("--top-p", type=float, help="Default top_p")
parser.add_argument("--top-k", type=int, help="Default top_k")

args = parser.parse_args()

config = load_config()

MODEL_NAME = config["model"]["name"]
MAX_TOKENS = config["model"]["max_tokens"]
HOST = config["server"]["host"]
PORT = config["server"]["port"]

DEFAULT_TEMPERATURE = config["model"].get("temperature", 1.0)
DEFAULT_TOP_P = config["model"].get("top_p", 1.0)
DEFAULT_TOP_K = config["model"].get("top_k", 50)

if args.model:
    MODEL_NAME = args.model

if args.port:
    PORT = args.port

if args.temperature:
    DEFAULT_TEMPERATURE = args.temperature

if args.top_p:
    DEFAULT_TOP_P = args.top_p

if args.top_k:
    DEFAULT_TOP_K = args.top_k


class LLMService(llm_pb2_grpc.LLMServiceServicer):

    def __init__(self):
        self.tokenizer, self.model = load_model(MODEL_NAME)

    def Generate(self, request, context):
        temperature = request.temperature if request.temperature > 0 else DEFAULT_TEMPERATURE
        top_p = request.top_p if request.top_p > 0 else DEFAULT_TOP_P
        top_k = request.top_k if request.top_k > 0 else DEFAULT_TOP_K

        for token in stream_tokens(
            request.text,
            self.tokenizer,
            self.model,
            max_tokens=MAX_TOKENS,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        ):
            yield llm_pb2.Token(text=token)

    def BatchGenerate(self, request, context):
        """Handle batch generation of prompts.

        Each incoming Prompt is assigned a unique ID and processed sequentially.
        The server streams back BatchResponse messages containing the ID and token.
        """
        for prompt_msg in request.prompts:
            prompt_id = str(uuid.uuid4())
            temperature = prompt_msg.temperature if prompt_msg.temperature > 0 else DEFAULT_TEMPERATURE
            top_p = prompt_msg.top_p if prompt_msg.top_p > 0 else DEFAULT_TOP_P
            top_k = prompt_msg.top_k if prompt_msg.top_k > 0 else DEFAULT_TOP_K

            for token in stream_tokens(
                prompt_msg.text,
                self.tokenizer,
                self.model,
                max_tokens=MAX_TOKENS,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ):
                yield llm_pb2.BatchResponse(id=prompt_id, token=token)


def serve():
    print("[BOOT] Starting gRPC server...")

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4)
    )

    llm_pb2_grpc.add_LLMServiceServicer_to_server(
        LLMService(),
        server
    )


    server.add_insecure_port(f"{HOST}:{PORT}")
    server.start()

    print("[READY] gRPC LLM server running on :50051")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
