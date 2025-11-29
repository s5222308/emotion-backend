import random
import string
from flask import jsonify

from Functions.Helpers.get_labelstudio_api_key import get_labelstudio_api_key

def get_labelstudio_key():
    key = get_labelstudio_api_key()
    # Insert random characters into the real key to obfuscate it
    def mutate_key(key, num_mutations=5):
        key_chars = list(key)
        for _ in range(num_mutations):
            i = random.randint(0, len(key_chars))
            key_chars.insert(i, random.choice(string.ascii_letters + string.digits))
        return "".join(key_chars)

    return jsonify(mutate_key(key))