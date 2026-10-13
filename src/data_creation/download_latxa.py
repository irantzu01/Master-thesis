from huggingface_hub import hf_hub_download

FOLDERS = ['euscrawl-v1.1', 'wikipedia', 'CulturaX', 'colossal-oscar', 'hplt-v1']
SPLITS = ['train', 'valid', 'test']

LOCAL_ROOT = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data/latxa-corpus-v1.1"

for folder in FOLDERS:
    for split in SPLITS:
        remote_filename = f"{folder}/{split}.jsonl.gz"
        print(f"[INFO] Downloading {remote_filename}...")
        local_path = hf_hub_download(
            repo_id="HiTZ/latxa-corpus-v1.1",
            filename=remote_filename,
            repo_type="dataset",
            local_dir=LOCAL_ROOT,
        )
        print(f"  -> saved to {local_path}")

print("[SUCCESS] All files downloaded.")