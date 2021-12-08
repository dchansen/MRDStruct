from unpacker import *
import numpy as np
import io

def main():
    data = {"Pie": np.zeros(20, dtype=np.complex64), "Penguins": 10}

    bytes = io.BytesIO()

    dump(data,bytes)
    bytes.seek(0)

    data2 = load(bytes)

    assert data == data2

if __name__ == "__main__":
    main()
