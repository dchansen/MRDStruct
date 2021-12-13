from dataclasses import dataclass
import dataclasses
from serialization import *
from typing import Dict, Optional
import numpy.typing as npt
import numpy as np
import io
import mrdstruct


@dataclass
class NoiseData:
    header: str
    noise_dwll_time_us: float
    noise_covariance_matrix: npt.NDArray[np.complex64]


def main():
    data = NoiseData("not actually a header", 10, np.ones((16, 16), dtype=np.complex64))
    buffer = io.BytesIO()
    mrdstruct.dump(data,buffer)
    buffer.seek(0)
    data2 = mrdstruct.load(buffer, data_type=NoiseData)


    for field in dataclasses.fields(data):
        val1 = getattr(data,field.name)
        val2 = getattr(data,field.name)
        if isinstance(val1,np.ndarray):
            assert np.array_equal(val1,val2)
        else:
            assert val1 == val2 



if __name__ == "__main__":
    main()