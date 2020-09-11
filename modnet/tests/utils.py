import hashlib


def get_sha512_of_file(fname):
    """ Returns the hexdigest of the SHA512 checksum of the
    file found at fname.

    """
    block_size = 65536
    _hash = hashlib.sha512()
    with open(fname, "rb") as f:
        fb = f.read(block_size)
        while fb:
            _hash.update(fb)
            fb = f.read(block_size)

    return _hash.hexdigest()
