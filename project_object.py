import json

class ProjectObject:

    def __init__(self, secret_path='secrets/secrets.json', config_path='config/config.json'):
        self._secrets = self._get_secrets(secret_path)
        self._config = self._get_config(config_path)

    def _get_secrets(self, secret_path):
        try:
            with open(secret_path) as f:
                return json.load(f)
        except Exception as e:
            print("Error: ", e)

    def _get_config(self, config_path):
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            print("Error: ", e)