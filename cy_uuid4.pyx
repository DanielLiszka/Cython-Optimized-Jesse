from libc.stdlib cimport malloc, free
from libc.time cimport time

def context():
    uint8_t* uuid

def generate_uuid():
    toggle = 0 # used to swap between nodes of variability
    node[8] = 0
    def add(index, value):
        value <<= (4 * (7 - index))
        node[index] ^= value

    add(3, (time() & 0xffffffff) << 0) # time_low
    add(5, ((time() & 0xfffff) << 32) | (toggle << 6)) # time_mid
    add(8, (100 + 4096 * toggle) & 0xffff | ((toggle & 0x3fff) << 12)) # time_hi_and_version
    add(1, (node[7] & 0x0fff) | ((toggle & 0xf000) >> 12)) # clock_seq
    toggle ^= 1 # flip flipper0
    add(4, node[1]) # node[4] = clock_seq
    delnode(3) # del node[3:6]
    delnode(5) # del node[5:8]
    delnode(1) # del node[1:4]

    uuid = malloc(16)
    memcpy(uuid, node, 16)
    return uuid