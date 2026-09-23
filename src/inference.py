import os
import argparse
import contextlib
import sys
import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import torch
from openai import AzureOpenAI

try:
    from vllm import LLM as VLLMClient
except ImportError:
    VLLMClient = None

try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    pipeline = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

from lib.utils import (
    all_at_once as gpt_all_at_once,
    step_by_step as gpt_step_by_step,
    binary_search as gpt_binary_search
)

from lib.local_model import (
    analyze_all_at_once_local,
    analyze_step_by_step_local,
    analyze_binary_search_local
)


KNOWN_GPT_MODELS = {"gpt-4o", "gpt4", "gpt4o-mini"}
LOCAL_LLAMA_ALIASES = {"llama-8b", "llama-70b", "llama-3.1-8B-Instruct", "llama-3.1-70B-Instruct", "llama-3B"}
LOCAL_QWEN_ALIASES = {"qwen-7b", "qwen-72b"}
LOCAL_MODEL_ALIASES = LOCAL_LLAMA_ALIASES | LOCAL_QWEN_ALIASES
ALL_MODELS = list(KNOWN_GPT_MODELS | LOCAL_MODEL_ALIASES)

LOCAL_MODEL_MAP = {
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "llama-3B": "meta-llama/Llama-3.2-3B",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
}


def resolve_local_model_id(model_alias: str, model_path: str = None) -> str:
    if model_path and model_path.strip():
        return model_path.strip()

    env_key = model_alias.upper().replace("-", "_") + "_PATH"
    env_value = os.getenv(env_key)
    if env_value and env_value.strip():
        return env_value.strip()

    env_value = os.getenv("LOCAL_MODEL_PATH")
    if env_value and env_value.strip():
        return env_value.strip()

    return LOCAL_MODEL_MAP[model_alias]


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Analyze multi-agent chat history using specific models.")

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["all_at_once", "step_by_step", "binary_search"],
        help="The analysis method to use."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=ALL_MODELS,
        help=f"Model identifier. Choose from: {', '.join(ALL_MODELS)}. If omitted, uses MODEL or LLAMA_8B_PATH-style environment defaults when available."
    )
    parser.add_argument(
        "--directory_path",
        type=str,
        default = "../Who&When/Algorithm-Generated",
        help="Path to the directory containing JSON chat history files. Default: '../Who&When/Algorithm-Generated'."
    )

    parser.add_argument(
        "--is_handcrafted",
        type=str,
        default="False",
        choices=['True', 'False'], # If you want to test Hand-Crafted, set is_handcrafted to be True.
        help="Specify 'True' or 'False'. Default: 'False'."
    )


    parser.add_argument(
        "--api_key", type=str, default= " ", #Please enter your api key here.
        help="Azure OpenAI API Key. Conditionally required for GPT models. Uses AZURE_OPENAI_API_KEY env var if available."
    )
    parser.add_argument(
        "--azure_endpoint", type=str, default=" ", #Please enter your azure_endpoint here.
        help="Azure OpenAI Endpoint URL. Conditionally required for GPT models. Uses AZURE_OPENAI_ENDPOINT env var if available."
    )
    parser.add_argument(
        "--api_version", type=str, default="2024-08-01-preview",
        help="Azure OpenAI API Version. Used only for GPT models."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024,
        help="Maximum number of tokens for GPT API response. Used only for GPT models."
    )

    parser.add_argument(
        "--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu",
        help="Device for local model inference (e.g., 'cuda', 'cuda:0', 'cpu'). Default: 'cuda' if available, else 'cpu'."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional local model path or Hugging Face repo ID to override the default alias mapping. Useful when a model is cached locally or access is restricted to a gated repo."
    )

    args = parser.parse_args()

    if args.model is None:
        env_model = os.getenv("MODEL")
        if env_model and env_model.strip():
            args.model = env_model.strip()
        else:
            for candidate in ["llama-8b", "qwen-7b", "llama-70b", "qwen-72b"]:
                if candidate in ALL_MODELS:
                    env_alias = candidate
                    break
            else:
                env_alias = None

            if env_alias is not None:
                args.model = env_alias

    if args.model is None:
        print("Error: --model is required unless MODEL or a default local alias is configured in the environment.")
        sys.exit(1)

    client_or_model_obj = None
    model_type = None # gpt, llama, qwen
    model_family = None 
    model_id_or_deployment = args.model

    if args.model in KNOWN_GPT_MODELS:
        model_type = 'gpt'
        model_family = 'gpt'
        print(f"Selected GPT model: {args.model}")
       
        if not args.api_key:
            print("Error: --api_key or AZURE_OPENAI_API_KEY environment variable is required for GPT models")
            sys.exit(1)
        if not args.azure_endpoint:
            print("Error: --azure_endpoint or AZURE_OPENAI_ENDPOINT environment variable is required for GPT models")
            sys.exit(1)
        try:
            client_or_model_obj = AzureOpenAI(
                api_key=args.api_key,
                api_version=args.api_version,
                azure_endpoint=args.azure_endpoint,
            )
            print(f"Successfully initialized AzureOpenAI client for endpoint: {args.azure_endpoint}")
        except Exception as e:
            print(f"Error initializing Azure OpenAI client: {e}")
            sys.exit(1)

    elif args.model in LOCAL_MODEL_ALIASES:
        model_type = 'local'
        model_id_or_deployment = resolve_local_model_id(args.model, args.model_path)

        if args.model in LOCAL_LLAMA_ALIASES:
            model_family = 'llama'
        elif args.model in LOCAL_QWEN_ALIASES:
            model_family = 'qwen'
        else:
            model_family = None

        print(f"Selected local model: {args.model} ({model_id_or_deployment}) on device {args.device}")

        try:
            if VLLMClient is not None:
                print(f"Initializing vLLM backend for {model_id_or_deployment}...")
                client_or_model_obj = VLLMClient(model=model_id_or_deployment, trust_remote_code=True)
                print(f"Successfully initialized vLLM model for {model_id_or_deployment}.")
            elif pipeline is not None:
                print(f"Initializing Hugging Face pipeline for {model_id_or_deployment}...")
                client_or_model_obj = pipeline(
                    "text-generation",
                    model=model_id_or_deployment,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device=args.device,
                )
                print(f"Successfully initialized Hugging Face pipeline on {args.device}.")
            else:
                print("Error: neither vLLM nor transformers are available for local inference.")
                sys.exit(1)
        except Exception as e:
            print(f"Error initializing local model for {model_id_or_deployment}: {e}")
            if VLLMClient is None:
                print("Make sure you have sufficient VRAM/RAM and necessary libraries (transformers, torch, accelerate).")
            sys.exit(1)
    else:
        print(f"Error: Invalid model '{args.model}' specified.")
        sys.exit(1)


    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    handcrafted_suffix = "_handcrafted" if args.is_handcrafted == "True" else "_alg_generated"
    output_filename = f"{args.method}_{args.model.replace('/','_')}{handcrafted_suffix}.txt"
    output_filepath = os.path.join(output_dir, output_filename)
    
    args.is_handcrafted = True if args.is_handcrafted == "True" else False # Update: Convert string to boolean

    print(f"Analysis method: {args.method}")
    print(f"Model Alias: {args.model} (Family: {model_family})")
    print(f"Output will be saved to: {output_filepath}")

    try:
        with open(output_filepath, 'w', encoding='utf-8') as output_file, contextlib.redirect_stdout(output_file):
            print(f"--- Starting Analysis: {args.method} ---")
            print(f"Timestamp: {datetime.datetime.now()}")
            print(f"Model Family: {model_family}")
            print(f"Model Used: {model_id_or_deployment}")
            print(f"Input Directory: {args.directory_path}")
            print(f"Is Handcrafted: {args.is_handcrafted}")
            print("-" * 20)

            if model_type == 'gpt':
                if args.method == "all_at_once":
                    gpt_all_at_once(
                        client=client_or_model_obj,
                        directory_path=args.directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model=args.model,
                        max_tokens=args.max_tokens
                    )
                elif args.method == "step_by_step":
                    gpt_step_by_step(
                        client=client_or_model_obj,
                        directory_path=args.directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model=args.model,
                        max_tokens=args.max_tokens
                    )
                elif args.method == "binary_search":
                    gpt_binary_search(
                        client=client_or_model_obj,
                        directory_path=args.directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model=args.model,
                        max_tokens=args.max_tokens
                    )
            elif model_type == 'local':
                if args.method == "all_at_once":
                    analyze_all_at_once_local(
                        model_obj=client_or_model_obj,
                        directory_path=args.directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model_family=model_family
                    )
                elif args.method == "step_by_step":
                    analyze_step_by_step_local(
                        model_obj=client_or_model_obj,
                        directory_path=args.directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model_family=model_family
                    )
                elif args.method == "binary_search":
                    analyze_binary_search_local(
                        model_obj=client_or_model_obj,
                        directory_path=args.directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model_family=model_family
                    )

            else:
                 print(f"Internal Error: Unknown model_type '{model_type}' during function call.")


            print("-" * 20)
            print(f"--- Analysis Complete ---")

        print(f"Analysis finished. Output saved to {output_filepath}")

    except Exception as e:
        print(f"\n!!! An error occurred during analysis or file writing: {e} !!!", file=sys.stderr)
  
if __name__ == "__main__":
    main()