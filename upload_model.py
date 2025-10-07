import kagglehub

#kagglehub.login()

# Replace with path to directory containing model files.
LOCAL_MODEL_DIR = '/mnt/nfs-mnj-hot-99/tmp/hokuyama/jigsaw-agile-community-rules/outputs/normal-Qwen3-4B-2025-10-05_20-48-29/checkpoint-5000'

MODEL_SLUG = 'jigzaw-qwen3' # Replace with model slug.

# Learn more about naming model variations at
# https://www.kaggle.com/docs/models#name-model.
VARIATION_SLUG = '4b-grpo' # Replace with variation slug.

kagglehub.model_upload(
  handle = f"hokuyama/{MODEL_SLUG}/transformers/{VARIATION_SLUG}",
  local_model_dir = LOCAL_MODEL_DIR,
  version_notes = 'normal-Qwen3-4B-2025-10-05_20-48-29-checkpoint-5000')
