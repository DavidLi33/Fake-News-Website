from huggingface_hub import upload_folder

# 👇 replace with your real username/repo on Hugging Face
repo_id = "davidli33/bertclassifierbest"

upload_folder(
    repo_id=repo_id,
    folder_path="./bertclassifierbest",
    commit_message="Upload BERT classifier"
)