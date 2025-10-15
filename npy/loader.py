from pyannote.database import ProtocolFile
from pathlib import Path
import numpy
from typing import Text
import traceback

class NpyLoader:
    def __init__(self, npy: Path):
        self.npy = npy
        

    def __call__(self, current_file: ProtocolFile) -> Text:
        uri = current_file["uri"]
        return self.npy
