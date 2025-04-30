import json
from typing import List, Dict

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath


    def load_json(self):
        """

        :param filepath:
        :return:
        """
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            assert isinstance(data, list)
            return data
        except FileNotFoundError:
            raise Exception(f"File not found:{self.filepath}")
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON format in file: {self.filepath}")
        except AssertionError as e:
            raise Exception(str(e))

