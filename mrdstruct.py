import msgpack
import numpy as np
import struct
import functools
import serialization
import dacite
import dataclasses
import ismrmrd

ismrmrd_types = {3: ismrmrd.AcquisitionHeader, 4: ismrmrd.ImageHeader, 5: ismrmrd.WaveformHeader}
ismrmrd_codes = {val: key for key, val in ismrmrd_types.items()}


def load(source, data_type=None):
    msg_buffer = serialization.read_byte_string(source)
    nitems = serialization.read_value(source, serialization.uint64)
    data_arrays = [serialization.read_typed_array(source) for n in range(nitems)]

    def ext_hook(code, data):
        if code == 2:
            buffer_index = serialization.uint64.unpack(data)[0]
            assert (buffer_index < len(data_arrays))
            return data_arrays[buffer_index]

        if code in ismrmrd_types:
            data_type = ismrmrd_types[code]
            return data_type.from_buffer_copy(data)

        return msgpack.ExtType(code, data)

    result = msgpack.unpackb(msg_buffer, ext_hook=ext_hook, raw=False)
    if data_type is not None:
        assert dataclasses.is_dataclass(data_type)
        return serialization.from_dict(result, data_type)
    return result


def dump(obj, destination):
    arrays = []

    def default(field):
        if dataclasses.is_dataclass(field):
            return dataclasses.asdict(field)
        if type(field) in ismrmrd_codes:
            code = ismrmrd_codes[type(field)]
            return msgpack.ExtType(code, bytes(field))

        if isinstance(field, np.ndarray):
            index = len(arrays)
            arrays.append(field)

            return msgpack.ExtType(2, serialization.uint64.pack(index))

        return TypeError("Not an ndarray")

    buffer = msgpack.dumps(obj, default=default, use_bin_type=True)
    destination.write(serialization.uint64.pack(len(buffer)))
    destination.write(buffer)

    nitems = len(arrays)
    destination.write(serialization.uint64.pack(nitems))
    for arr in arrays:
        serialization.write_typed_array(destination, arr)
