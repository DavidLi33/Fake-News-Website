from huggingface_hub import upload_folder

repo_id = "davidli33/bertclassifierbest"

upload_folder(
    repo_id=repo_id,
    folder_path="./bertclassifierbest",
    commit_message="Upload BERT classifier"
)