import grpc
import sys
import argparse
from google.protobuf.timestamp_pb2 import Timestamp

from generated import llm_pb2
from generated import llm_pb2_grpc

def batch_generate(prompts, temperature=1.0, top_p=1.0, top_k=50):
    """Send multiple prompts and collect filtered tokens by ID"""
    request = llm_pb2.BatchRequest(
        prompts=[llm_pb2.Prompt(
            text=prompt_text,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        ) for prompt_text in prompts]
    )
    response_stream = stub.BatchGenerate(request)
    results = []
    for batch_resp in response_stream:
        results.append({
            'id': batch_resp.id,
            'token': batch_resp.token
        })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="*", default=["Hello world"], help="Input prompt(s) (multiple can be provided)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    channel = grpc.insecure_channel("localhost:50051")
    global stub
    stub = llm_pb2_grpc.LLMServiceStub(channel)

    if len(args.prompt) == 1:
        response = stub.Generate(
            llm_pb2.Prompt(
                text=args.prompt[0],
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k
            )
        )
        for token in response:
            print(token.text, end="", flush=True)
    else:
        batch_responses = batch_generate(
            args.prompt,
            args.temperature,
            args.top_p,
            args.top_k
        )
        print("\nBatch Responses (ID:token pairs):")
        for resp in batch_responses:
            print(f"ID:{resp['id']} -> {resp['token']}")

if __name__ == "__main__":
    main()