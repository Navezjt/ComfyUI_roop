import tempfile

def convert_to_sd(img):
    return [False, tempfile.NamedTemporaryFile(delete=False, suffix=".png")]
