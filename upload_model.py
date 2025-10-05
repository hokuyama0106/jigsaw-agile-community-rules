import kagglehub

#kagglehub.login()

# Replace with path to directory containing model files.
LOCAL_MODEL_DIR = '/mnt/nfs-mnj-hot-99/tmp/hokuyama/make-data-count/outputs/Qwen2.5-14B-Instruct-merged-7-awq'

MODEL_SLUG = 'mdc-qwen' # Replace with model slug.

# Learn more about naming model variations at
# https://www.kaggle.com/docs/models#name-model.
VARIATION_SLUG = '14b-instruct-grpo-awq' # Replace with variation slug.

kagglehub.model_upload(
  handle = f"hokuyama/{MODEL_SLUG}/transformers/{VARIATION_SLUG}",
  local_model_dir = LOCAL_MODEL_DIR,
  version_notes = 'Qwen2.5-14B-Instruct-merged-7-awq')
