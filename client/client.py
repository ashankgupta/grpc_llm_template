import grpc
import sys
import argparse

from generated import llm_pb2
from generated import llm_pb2_grpc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="Hello world")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    channel = grpc.insecure_channel("localhost:50051")
    stub = llm_pb2_grpc.LLMServiceStub(channel)

    response = stub.Generate(
        llm_pb2.Prompt(
            text=args.prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )
    )

    for token in response:
        print(token.text, end="", flush=True)


if __name__ == "__main__":
    main()
