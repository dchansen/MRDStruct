import msgpack
import numpy as np
import struct
import functools
import serialization


def load(source):
    msg_buffer = serialization.read_byte_string(source)
    nitems = serialization.read(source, serialization.uint64)
    data_arrays = [serialization.read_typed_array(source) for n in range(nitems)]

    def ext_hook(code,data):
        if code == 2:
            buffer_index= serialization.uint64.unpack(data)[0]
            assert(buffer_index < len(data_arrays))
            return data_arrays[buffer_index]
        return msgpack.ExtType(code,data)

    return msgpack.unpackb(msg_buffer,ext_hook=ext_hook,raw=False)

def dump(obj,destination):


    arrays = []
    def default(field):
        if isinstance(field,np.ndarray):
            index = len(arrays)
            arrays.append(field)

            return msgpack.ExtType(2, serialization.uint64.pack(index))

        return TypeError("Not an ndarray")

    buffer = msgpack.dumps(obj,default=default, use_bin_type=True)
    destination.write(serialization.uint64.pack(len(buffer)))
    destination.write(buffer)

    nitems = len(arrays)
    destination.write(serialization.uint64.pack(nitems))
    for arr in  arrays:
        serialization.write_typed_array(destination, arr)









