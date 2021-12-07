import msgpack
import numpy as np
import struct
import functools

Bool = struct.Struct('<?')
uint32 = struct.Struct('<I')
uint64 = struct.Struct('<Q')
uint16 = struct.Struct('<H')

typemap = {
    1: np.float32,
    2: np.float64,
    3: np.uint16,
    4: np.uint32,
    5: np.uint64,
    6: np.int16,
    7: np.int32,
    8: np.int64,
    9: np.complex32,
    10: np.complex64
}

def read(source, type):
    return type.unpack(source.read(type.size))[0]

def read_vector(source, np_type=np.uint64):
    size = read(source, uint64)
    dtype = np.dtype(np_type)
    return np.frombuffer(source.read(size * dtype.itemsize), dtype=dtype)

def read_array(source, np_type=np.uint64):
    dtype = np.dtype(np_type)
    dimensions = read_vector(source)
    elements = int(functools.reduce(lambda a, b: a * b, dimensions))
    return np.reshape(np.frombuffer(source.read(elements * dtype.itemsize), dtype=dtype), dimensions, order='F')

def read_byte_string(source):
    length = read(source,uint64)
    byte_string = source.read(length)
    return byte_string

def read_type(source):
    return typemap[read(source, uint16)]

def read_typed_array(source):
    obj_type = read_type(source)
    return read_array(source, obj_type)

def loads(source):
    msg_buffer = read_byte_string(source)
    nitems = read(source,uint64)
    data_arrays = [ read_typed_array(source) for n in range(nitems)]

    def ext_hook(code,data):
        if code == 2:
            buffer_index= uint64.unpack(data)
            assert(buffer_index < len(data_arrays))
            return data_arrays[buffer_index]
        return msgpack.ExtType(code,data)

    return msgpack.unpackb(msg_buffer,ext_hook=ext_hook,raw=False)










