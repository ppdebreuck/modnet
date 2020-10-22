import hashlib


def get_hash_of_file(fname, algo="sha512"):
    """ Returns the hexdigest of the SHA512 checksum of the
    file found at fname.

    """
    block_size = 65536
    if algo.lower() == "md5":
        _hash = hashlib.md5()
    else:
        _hash = hashlib.sha512()
    with open(fname, "rb") as f:
        fb = f.read(block_size)
        while fb:
            _hash.update(fb)
            fb = f.read(block_size)

    return _hash.hexdigest()
