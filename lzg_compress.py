#!/usr/bin/env python3
"""
Compress directory contents into LZG format (liblzg v1.0.10 compatible).
Creates a tar archive of the directory and compresses it with the LZG algorithm.
"""

import struct
import os
import tarfile
import io
import sys

LZG_HEADER_SIZE = 16
LZG_METHOD_COPY = 0
LZG_METHOD_LZG1 = 1

_LZG_LENGTH_DECODE_LUT = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 35, 48, 72, 128
]

_LENGTH_TO_INDEX = {}
for _tl in range(2, 256):
    _best_idx, _best_val = -1, 0
    for _idx, _val in enumerate(_LZG_LENGTH_DECODE_LUT):
        if _val <= _tl and _val > _best_val:
            _best_val = _val
            _best_idx = _idx
    if _best_idx >= 0:
        _LENGTH_TO_INDEX[_tl] = (_best_idx, _best_val)


def _calc_checksum(data):
    a, b = 1, 0
    for bv in data:
        a = (a + bv) & 0xFFFF
        b = (b + a) & 0xFFFF
    return (b << 16) | a


def _find_markers(data):
    freq = [0] * 256
    for bv in data:
        freq[bv] += 1
    return tuple(sorted(range(256), key=lambda x: (freq[x], x))[:4])


def _hash3(data, pos):
    return ((data[pos] << 16) ^ (data[pos + 1] << 8) ^ data[pos + 2]) & 0x3FFFF


def lzg_compress(data):
    n = len(data)
    if n == 0:
        encoded = b''
        cs = _calc_checksum(encoded)
        return b'LZG' + struct.pack('>III', 0, 0, cs) + bytes([LZG_METHOD_COPY])

    data = bytes(data)
    m1, m2, m3, m4 = _find_markers(data)
    marker_set = {m1, m2, m3, m4}

    encoded = bytearray([m1, m2, m3, m4])

    HASH_SIZE = 1 << 18
    HASH_MASK = HASH_SIZE - 1
    hash_head = [-1] * HASH_SIZE
    hash_prev = [-1] * n

    MAX_CHAIN = 32
    MAX_OFFSET = 526343
    GOOD_ENOUGH = 48

    i = 0
    last_pct = -1
    while i < n:
        pct = i * 100 // n
        if pct != last_pct and pct % 10 == 0:
            print(f"  Compressing... {pct}%", end='\r')
            last_pct = pct

        best_len = 1
        best_off = 0

        if i + 2 < n:
            h = _hash3(data, i)
            j = hash_head[h]
            chain = 0
            while j >= 0 and chain < MAX_CHAIN:
                off = i - j
                if off > MAX_OFFSET or off < 1:
                    break
                if data[j] == data[i] and data[j + 1] == data[i + 1] and data[j + 2] == data[i + 2]:
                    max_ml = min(n - i, 128, n - j)
                    ml = 3
                    while ml < max_ml and data[j + ml] == data[i + ml]:
                        ml += 1
                    if ml > best_len:
                        best_len = ml
                        best_off = off
                        if best_len >= GOOD_ENOUGH:
                            break
                j = hash_prev[j]
                chain += 1

        emitted = False
        if best_len >= 2 and best_off >= 1:
            offset = best_off
            best_savings = 0
            best_cmd = None
            best_actual_len = 0

            capped = min(best_len, 128)
            if capped in _LENGTH_TO_INDEX:
                lut_idx, lut_len = _LENGTH_TO_INDEX[capped]
            else:
                lut_idx, lut_len = -1, 0

            if 1 <= offset <= 8 and lut_idx >= 0 and lut_len >= 2:
                b1 = ((offset - 1) << 5) | lut_idx
                if b1 != 0:
                    s = lut_len - 2
                    if s > best_savings:
                        best_savings = s
                        best_cmd = bytes([m4, b1])
                        best_actual_len = lut_len

            if 8 <= offset <= 71 and best_len >= 3:
                al = min(best_len, 6)
                b1 = ((al - 3) << 6) | (offset - 8)
                if b1 != 0:
                    s = al - 2
                    if s > best_savings:
                        best_savings = s
                        best_cmd = bytes([m3, b1])
                        best_actual_len = al

            if 8 <= offset <= 2055 and lut_idx >= 0 and lut_len >= 2:
                oadj = offset - 8
                b1 = (((oadj >> 8) & 0x07) << 5) | lut_idx
                if b1 != 0:
                    s = lut_len - 3
                    if s > best_savings:
                        best_savings = s
                        best_cmd = bytes([m2, b1, oadj & 0xFF])
                        best_actual_len = lut_len

            if 2056 <= offset <= MAX_OFFSET and lut_idx >= 0 and lut_len >= 2:
                oadj = offset - 2056
                b1 = (((oadj >> 16) & 0x07) << 5) | lut_idx
                if b1 != 0:
                    s = lut_len - 4
                    if s > best_savings:
                        best_savings = s
                        best_cmd = bytes([m1, b1, (oadj >> 8) & 0xFF, oadj & 0xFF])
                        best_actual_len = lut_len

            if best_cmd is not None and best_savings > 0:
                encoded.extend(best_cmd)
                for k in range(best_actual_len):
                    pos = i + k
                    if pos + 2 < n:
                        hk = _hash3(data, pos)
                        hash_prev[pos] = hash_head[hk]
                        hash_head[hk] = pos
                i += best_actual_len
                emitted = True

        if not emitted:
            bv = data[i]
            encoded.append(bv)
            if bv in marker_set:
                encoded.append(0)
            if i + 2 < n:
                h = _hash3(data, i)
                hash_prev[i] = hash_head[h]
                hash_head[h] = i
            i += 1

    print("  Compressing... 100%")

    encoded_bytes = bytes(encoded)
    if len(encoded_bytes) >= n:
        cs = _calc_checksum(data)
        return b'LZG' + struct.pack('>III', n, n, cs) + bytes([LZG_METHOD_COPY]) + data

    cs = _calc_checksum(encoded_bytes)
    header = b'LZG' + struct.pack('>III', n, len(encoded_bytes), cs) + bytes([LZG_METHOD_LZG1])
    return header + encoded_bytes


def lzg_decompress(data):
    if len(data) < LZG_HEADER_SIZE:
        raise ValueError("Input too small")
    if data[0:3] != b'LZG':
        raise ValueError("Bad magic")

    decoded_size = struct.unpack('>I', data[3:7])[0]
    encoded_size = struct.unpack('>I', data[7:11])[0]
    checksum = struct.unpack('>I', data[11:15])[0]
    method = data[15]

    if encoded_size != len(data) - LZG_HEADER_SIZE:
        raise ValueError(f"Encoded size mismatch: header says {encoded_size}, actual {len(data) - LZG_HEADER_SIZE}")

    enc = data[LZG_HEADER_SIZE:]
    if _calc_checksum(enc) != checksum:
        raise ValueError("Checksum mismatch")

    if method == LZG_METHOD_COPY:
        if decoded_size != encoded_size:
            raise ValueError("COPY size mismatch")
        return enc

    if method != LZG_METHOD_LZG1:
        raise ValueError(f"Unknown method {method}")

    src = memoryview(enc)
    si = 0
    out = bytearray(decoded_size)
    di = 0

    marker1, marker2, marker3, marker4 = src[0], src[1], src[2], src[3]
    si = 4

    is_marker = [False] * 256
    is_marker[marker1] = True
    is_marker[marker2] = True
    is_marker[marker3] = True
    is_marker[marker4] = True

    while si < len(src):
        symbol = src[si]; si += 1
        if not is_marker[symbol]:
            out[di] = symbol; di += 1
        else:
            b = src[si]; si += 1
            if b:
                if symbol == marker1:
                    length = _LZG_LENGTH_DECODE_LUT[b & 0x1f]
                    b2 = src[si]; si += 1
                    offset = ((b & 0xe0) << 11) | (b2 << 8) | src[si]; si += 1
                    offset += 2056
                elif symbol == marker2:
                    length = _LZG_LENGTH_DECODE_LUT[b & 0x1f]
                    b2 = src[si]; si += 1
                    offset = ((b & 0xe0) << 3) | b2
                    offset += 8
                elif symbol == marker3:
                    length = (b >> 6) + 3
                    offset = (b & 0x3f) + 8
                else:
                    length = _LZG_LENGTH_DECODE_LUT[b & 0x1f]
                    offset = (b >> 5) + 1

                cp = di - offset
                for k in range(length):
                    out[di] = out[cp + k]; di += 1
            else:
                out[di] = symbol; di += 1

    if di != decoded_size:
        raise ValueError(f"Decoded size mismatch: expected {decoded_size}, got {di}")
    return bytes(out)


def create_tar(input_dir):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.USTAR_FORMAT) as tar:
        for fname in sorted(os.listdir(input_dir)):
            fpath = os.path.join(input_dir, fname)
            if os.path.isfile(fpath):
                tar.add(fpath, arcname=fname)
    return buf.getvalue()


def compress(input_dir, output_file):
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    print(f"Creating tar archive of {input_dir}...")
    tar_data = create_tar(input_dir)
    print(f"  Tar size: {len(tar_data)} bytes")

    print("Compressing with LZG algorithm...")
    compressed = lzg_compress(tar_data)

    with open(output_file, 'wb') as f:
        f.write(compressed)

    ratio = len(compressed) / len(tar_data) * 100
    print(f"  Result: {len(tar_data)} -> {len(compressed)} bytes ({ratio:.1f}%)")
    print(f"  Output: {output_file}")

    print("Verifying by decompressing...")
    decompressed = lzg_decompress(compressed)
    if decompressed == tar_data:
        print("  Verification OK: decompressed data matches original.")
    else:
        print("  ERROR: decompressed data does NOT match!")
        sys.exit(1)


def extract(input_file, output_dir):
    if not os.path.isfile(input_file):
        print(f"Error: {input_file} not found")
        sys.exit(1)

    print(f"Reading {input_file}...")
    with open(input_file, 'rb') as f:
        compressed = f.read()
    print(f"  Compressed size: {len(compressed)} bytes")

    print("Decompressing LZG...")
    tar_data = lzg_decompress(compressed)
    print(f"  Decompressed size: {len(tar_data)} bytes")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Extracting files to {output_dir}...")
    tar_buf = io.BytesIO(tar_data)
    with tarfile.open(fileobj=tar_buf, mode='r') as tar:
        members = tar.getmembers()
        for member in members:
            if member.isfile():
                tar.extract(member, path=output_dir)
                print(f"  {member.name} ({member.size} bytes)")

    print(f"Done. Extracted {len(members)} files to {output_dir}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage:")
        print(f"  Compress:  {sys.argv[0]} compress <input_directory> <output.lzg>")
        print(f"  Extract:   {sys.argv[0]} extract <input.lzg> <output_directory>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "compress":
        if len(sys.argv) < 4:
            print(f"Usage: {sys.argv[0]} compress <input_directory> <output.lzg>")
            sys.exit(1)
        compress(sys.argv[2], sys.argv[3])

    elif cmd == "extract":
        if len(sys.argv) < 4:
            print(f"Usage: {sys.argv[0]} extract <input.lzg> <output_directory>")
            sys.exit(1)
        extract(sys.argv[2], sys.argv[3])

    else:
        print(f"Unknown command: {cmd}")
        print(f"Use 'compress' or 'extract'")
        sys.exit(1)


if __name__ == '__main__':
    main()
