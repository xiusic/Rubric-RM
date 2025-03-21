from transformers import AutoModel, AutoTokenizer

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_hub_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    api.delete_repo(args.hf_hub_path)
    api.create_repo(repo_id=args.hf_hub_path, private=False, exist_ok=True)
    api.upload_folder(folder_path=args.model_path, repo_id=args.hf_hub_path, repo_type="model")

if __name__ == "__main__":
    main()