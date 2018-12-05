from trees import *
from vl_codes import *
import arithmetic
import contextual_arithmetic
from itertools import groupby
from json import dump
from sys import argv
import json


def camzip(method, message_filename, context_chars=1, cond_prob_dict_filename=' ', cum_prob_dict_filename=' '):

    if not method == 'contextual arithmetic':
        with open(message_filename, 'rb') as fin:
            x = fin.read()

        frequencies = dict([(key, len(list(group))) for key, group in groupby(sorted(x))])
        n = sum([frequencies[a] for a in frequencies])
        p = dict([(a,frequencies[a]/n) for a in frequencies])

        if method == 'huffman' or method == 'shannon_fano':
            if (method == 'huffman'):
                xt = huffman(p)
                c = xtree2code(xt)
            else:
                c = shannon_fano(p)
                xt = code2xtree(c)

            y = vl_encode(x, c)

        elif method == 'arithmetic':
            y = arithmetic.encode(x,p)
        else:
            raise NameError('Compression method %s unknown' % method)

        y = bytes(bits2bytes(y))

        outfile = message_filename + '.cz' + method[0]

        with open(outfile, 'wb') as fout:
            fout.write(y)

        pfile = message_filename + '.czp'
        n = len(x)

        with open(pfile, 'w') as fp:
            dump(frequencies, fp)

    else:
        with open('cond_prob_models/' + cond_prob_dict_filename, 'r') as cond_prob_file:
            context_dict = json.load(cond_prob_file)
        with open('cond_prob_models/' + cum_prob_dict_filename, 'r') as cum_prob_file:
            cumulative_dict = json.load(cum_prob_file)
        with open('text_files/' + message_filename, 'r', encoding='utf-8-sig') as file:
            original_message = file.read()
        with open('encoded_messages/' + message_filename + '_zipped.cz', 'w') as zipped_file:
            zipped_message = bytes(bits2bytes(contextual_arithmetic.encode(original_message, context_dict, cumulative_dict, context_chars)))
            zipped_file.write(str(zipped_message))


if __name__ == "__main__":
    if (len(argv) != 3):
        print('Usage: python %s compression_method filename\n' % argv[0])
        print('Example: python %s huffman hamlet.txt' % argv[0])
        print('or:      python %s shannon_fano hamlet.txt' % argv[0])
        print('or:      python %s arithmetic hamlet.txt' % argv[0])
        exit()

    camzip(argv[1], argv[2])

