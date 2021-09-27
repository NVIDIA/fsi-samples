"""
 ////////////////////////////////////////////////////////////////////////////
 //
 // Copyright (C) NVIDIA Corporation.  All rights reserved.
 //
 // NVIDIA Sample Code
 //
 // Please refer to the NVIDIA end user license agreement (EULA) associated
 // with this source code for terms and conditions that govern your use of
 // this software. Any use, reproduction, disclosure, or distribution of
 // this software and related documentation outside the terms of the EULA
 // is strictly prohibited.
 //
 ////////////////////////////////////////////////////////////////////////////
"""


def get_data(data):
    total = 0
    for i, num in enumerate(data):
        total += num * 256**i
    return total


def set_data(content, data):
    length = len(content)
    for i in range(length-1, -1, -1):
        content[i] = data // (256**i)
        data = data % (256**i)
    return content


def examine_wav(content):
    print('header type:', content[0:4])
    print('file size: %d' % get_data(content[4:8]))
    print('file type header: %s' % content[8:12])
    print('file format chunk marker: %s' % content[12:16])
    print('format data length: %d, has to be 16' % get_data(content[16:20]))
    print('Type of format %d, 1 for pcm' % get_data(content[20:22]))
    print('Number of channels %d' % get_data(content[22:24]))
    print('Sample rate: %d' % get_data(content[24:28]))
    print('Byte rate: %d' % get_data(content[28:32]))
    print('Byte Per Sample * Channels : %d' % get_data(content[32:34]))
    print('Bits Per Sample: %d' % get_data(content[34:36]))
    print('data chunk header: %s' % content[36:40])
    print('data chunk size: %d' % get_data(content[40:44]))
    print(len(content), 'match', 44 + get_data(content[40:44]))


def add_header(newdata, bits_per_sample, channel, sr):
    n = bytearray(
        b'RIFF\xc4P\x05\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\xa0P\x05\x00'  # noqa
    )
    n[22:24] = set_data(n[22:24], channel)
    n[34:36] = set_data(n[34:36], bits_per_sample)
    n[32:34] = set_data(n[32:34], bits_per_sample // 8 * channel)
    n[24:28] = set_data(n[24:28], sr)
    n[28:32] = set_data(n[28:32], sr * bits_per_sample * channel // 8)
    n[40:44] = set_data(n[40:44], len(newdata))
    n[4:8] = set_data(n[4:8], 44 + len(newdata) - 8)
    return n + newdata
