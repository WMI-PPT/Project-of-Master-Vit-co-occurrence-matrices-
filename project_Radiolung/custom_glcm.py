import numpy as np
from radiomics import glcm
import logging

class CustomGLCM(glcm.RadiomicsGLCM):
    def __init__(self, inputImage, inputMask, **kwargs):
        super().__init__(inputImage, inputMask, **kwargs)
        self.glcm_raw_matrix = None

    def _calculateMatrix(self, voxelCoordinates=None):
        result = super()._calculateMatrix(voxelCoordinates=voxelCoordinates)
        if isinstance(result, np.ndarray):
            self.glcm_raw_matrix = result.copy()
        else:
            logging.warning("GLCM matrix calculation failed or returned unexpected type.")
        return result

