from mrdstruct import *
import mrdstruct
import numpy as np
import io
import ismrmrd


def main():
    data = {"Pie": np.zeros(20, dtype=np.complex64), "Penguins": 10, "header": ismrmrd.AcquisitionHeader(version=2),
            "acq_headers": np.array([ismrmrd.ImageHeader(), ismrmrd.ImageHeader()], dtype=object)}

    bytes = io.BytesIO()

    mrdstruct.dump(data, bytes)
    bytes.seek(0)

    data2 = mrdstruct.load(bytes)

    assert data.keys() == data2.keys()
    for key in data:
        if isinstance(data[key], np.ndarray):
            assert np.array_equal(data[key], data2[key])
        else:
            assert data[key] == data2[key]


if __name__ == "__main__":
    main()
