from trees import *
from vl_codes import *
import arithmetic
import contextual_arithmetic
from json import load
from sys import argv, exit
import json


def camunzip(filename, cond_prob_dict_filename=' ', cum_prob_dict_filename=' ', orig_message_filename=' '):
    if not filename[-1] == 'z':
        if filename[-1] == 'h':
            method = 'huffman'
        elif filename[-1] == 's':
            method = 'shannon_fano'
        elif filename[-1] == 'a':
            method = 'arithmetic'
        else:
            raise NameError('Unknown compression method')

        with open(filename, 'rb') as fin:
            y = fin.read()
        y = bytes2bits(y)

        pfile = filename[:-1] + 'p'
        with open(pfile, 'r') as fp:
            frequencies = load(fp)
        n = sum([frequencies[a] for a in frequencies])
        p = dict([(int(a),frequencies[a]/n) for a in frequencies])

        if method == 'huffman' or method == 'shannon_fano':
            if method == 'huffman':
                xt = huffman(p)
                c = xtree2code(xt)
            else:
                c = shannon_fano(p)
                xt = code2xtree(c)

            x = vl_decode(y, xt)

        elif method == 'arithmetic':
            x = arithmetic.decode(y,p,n)

        else:
            raise NameError('This will never happen (famous last words)')

        # '.cuz' for Cam UnZipped (don't want to overwrite the original file...)
        outfile = filename[:-4] + '.cuz'



    else:

        with open('encoded_messages/' + filename + '_zipped.cz', 'r') as zipped_file:
            with open('cond_prob_models/' + cond_prob_dict_filename, 'r') as cond_prob_file:
                context_dict = json.load(cond_prob_file)
            with open('cond_prob_models/' + cum_prob_dict_filename, 'r') as cum_prob_file:
                cumulative_dict = json.load(cum_prob_file)
            with open('text_files/' + orig_message_filename, 'r', encoding='utf-8-sig') as file:
                original_message = file.read()
            x = contextual_arithmetic.decode(bytes2bits(zipped_file.read()), context_dict, cumulative_dict,
                                                                len(original_message))

    with open(outfile, 'wb') as fout:
        fout.write(bytes(x))


if __name__ == "__main__":
    if len(argv) != 2:
        print('Usage: python %s filename\n' % argv[0])
        print('Example: python %s hamlet.txt.czh' % argv[0])
        print('or:      python %s hamlet.txt.czs' % argv[0])
        print('or:      python %s hamlet.txt.cza' % argv[0])
        exit()

    camunzip(argv[1])
