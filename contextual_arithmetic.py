from itertools import groupby
import json
from typing import Dict
from math import floor, ceil
from sys import stdout as so
from bisect import bisect
from arithmetic import encode_rescale, decode_rescale, make_cumulative_dict
from vl_codes import bits2bytes, bytes2bits


def write_training_data(*filenames):

    train_data = ''

    for filename in filenames:
        with open('text_files/' + filename, 'r') as file:
            train_data += file.read()

    with open("text_files/training_data.txt", "w+") as f:
        f.write(train_data)


def build_contextual_dict(prob_cont_dict_filename, cum_prob_dict_filename, context_char_no=1,
                          training_data_filename='training_data.txt'):
    if context_char_no > 4:
        raise ValueError("Due to how this process scales we are only allowing up to 4 context characters")

    with open("text_files/" + training_data_filename, "r", encoding='utf-8-sig') as training_file:
        training_data = training_file.read()

    frequencies = dict([(key, len(list(group))) for key, group in groupby(sorted(training_data))])
    Nin = sum([frequencies[a] for a in frequencies])
    prob_no_context = dict([(a, frequencies[a]/Nin) for a in frequencies])

    char_list = [character for character in frequencies]

    # If more than one context character then there are many permutations possible
    context_character_list = char_list
    for i in range(context_char_no - 1):
        prev_char_list = context_character_list
        context_character_list = []
        for element in prev_char_list:
            for character in char_list:
                context_character_list.append(element + character)

    # Initialise as empty dictionary
    contextual_dict = dict([(context_elements, dict([(character, 0) for character in char_list])) for context_elements in context_character_list])

    # Count frequency of occurrences of characters in training data given certain contexts
    for count, character in enumerate(training_data):
        if count > context_char_no:
            context_characters = ''
            for cont_count in range(context_char_no):
                ref = cont_count - context_char_no
                context_characters += training_data[count + ref]
            contextual_dict[context_characters][character] += 1

        else:
            pass

    # Remove any contexts that do not occur and divide each frequency by the total given that condition to get probs
    prob_cont_dict = {}
    for context_chars in contextual_dict:
        Nin = sum([contextual_dict[context_chars][char] for char in char_list])
        if not Nin == 0:
            prob_cont_dict[context_chars] = dict([(char, contextual_dict[context_chars][char]/Nin)
                                                  for char in char_list if contextual_dict[context_chars][char] > 0])

    # Write cumulative and normal conditional probability dictionaries to .json files to keep them permanently
    prob_cont_dict['no_context'] = prob_no_context

    with open('cond_prob_models/' + prob_cont_dict_filename, 'w') as fp:
        json.dump(prob_cont_dict, fp, sort_keys=True, indent=4)

    dict_of_cumulatives = dict([(context, make_cumulative_dict(prob_cont_dict[context])) for context in prob_cont_dict])
    with open('cond_prob_models/' + cum_prob_dict_filename, 'w') as fp:
        json.dump(dict_of_cumulatives, fp, sort_keys=True, indent=4)


def encode(input_message: str, cond_prob_dict: Dict, cumulative_prob_dict: Dict, context_char_no: int):
    """"Encodes input_message using probabilities char_probs"""
    precision = 32
    one = int(2 ** precision - 1)
    quarter = int(ceil(one / 4))

    compressed_message, lo, hi, straddle = [], 0, one, 0

    for char_counter, msg_char in enumerate(input_message):  # for every symbol

        # Progress bar
        if char_counter % 100 == 0:
            so.flush()
            so.write('Arithmetic encoded %d%%    \r' % int(floor(char_counter / len(input_message) * 100)))

        # When enough context exists to use full dictionary
        context_characters = ''
        if char_counter >= context_char_no:
            for cont_count in range(context_char_no):
                ref = cont_count - context_char_no
                context_characters += input_message[char_counter + ref]
        # Before there is sufficient context
        else:
            context_characters = 'no_context'

        char_probs = cond_prob_dict[context_characters]
        cum_prob_dict = cumulative_prob_dict[context_characters]

        lohi_range = hi - lo + 1  # The added 1 is necessary to avoid rounding issues

        # 2) narrow the interval end-points [lo,hi) to the new range [f,f+p]
        # within the old interval [lo,hi], being careful to round 'innwards'
        lo += ceil(cum_prob_dict[msg_char] * lohi_range)
        hi = lo + floor(char_probs[msg_char] * lohi_range)

        if lo == hi:
            raise NameError('Zero interval!')

        lo, hi, compressed_message, straddle = encode_rescale(lo, hi, compressed_message, one, straddle)

    # termination bits - after processing all input symbols, flush any bits still in the 'straddle' pipeline
    straddle += 1  # add 1 to straddle for "good measure" (ensures prefix-freeness)
    if lo < quarter:  # the position of lo determines the dyadic interval that fits
        compressed_message.append(0)
        compressed_message += [1 for s in range(straddle)]
    else:
        compressed_message.append(1)
        compressed_message += [0 for s in range(straddle)]

    return compressed_message


def decode(encoded_message: str, cond_prob_dict: Dict, cumulative_prob_dict: Dict,
           num_chars: int, context_char_no: int):
    precision = 32
    one = int(2 ** precision - 1)

    encoded_message.extend(precision * [0])  # dummy zeros to prevent index out of bound errors
    input_message = num_chars * [0]  # initialise all zeros

    # initialise by taking first 'precision' bits from y and converting to a number
    value = int(''.join(str(a) for a in encoded_message[0:precision]), 2)
    position = precision  # position where currently reading y
    lo, hi = 0, one

    for char_counter in range(num_chars):

        # When enough context exists to use full dictionary
        context_characters = ''
        if char_counter >= context_char_no:
            for cont_count in range(context_char_no):
                ref = cont_count - context_char_no
                context_characters += input_message[char_counter + ref]
        # Before there is sufficient context
        else:
            context_characters = 'no_context'

        char_probs = list(cond_prob_dict[context_characters].values())
        alphabet = list(cond_prob_dict[context_characters])
        cum_prob = list(cumulative_prob_dict[context_characters].values())

        if char_counter % 100 == 0:
            so.flush()
            so.write('Arithmetic decoded %d%%    \r' % int(floor(char_counter / num_chars * 100)))

        lohi_range = hi - lo + 1
        # This is an essential subtelty: the slowest part of the decoder is figuring out
        # which symbol lands us in an interval that contains the encoded binary string.
        # This can be extremely wasteful (o(n) where n is the alphabet size) if you proceed
        # by simple looping and comparing. Here we use Python's "bisect" function that
        # implements a binary search and is 100 times more efficient. Try
        # for a = [a for a in f if f[a]<(value-lo)/lohi_range)][-1] for a MUCH slower solution.
        a = bisect(cum_prob, (value - lo) / lohi_range) - 1
        input_message[char_counter] = alphabet[a]

        lo = lo + int(ceil(cum_prob[a] * lohi_range))
        hi = lo + int(floor(char_probs[a] * lohi_range))

        if lo == hi:
            raise NameError('Zero interval!')

        lo, hi, position, value = decode_rescale(lo, hi, encoded_message, position, one, value)

    return input_message


def main():
    # UNCOMMENT IF CHANGING TRAINING DATA
    write_training_data('hamlet.txt', 'war_and_peace.txt', 'encyclopedia_britannica.txt', 'romeo_and_juliet.txt',
                        'proj_gutenberg_encyclopedia.txt', 'great_expectations.txt')

    context_chars = 2
    cond_prob_dict_filename = 'cond_prob_dict' + str(context_chars) + '.json'
    cum_prob_dict_filename = 'cum_prob_dict' + str(context_chars) + '.json'
    training_message_filename = 'training_data.txt'
    message_filename = 'war_and_peace.txt'

    build_contextual_dict(cond_prob_dict_filename, cum_prob_dict_filename, context_chars, training_message_filename)

    with open('cond_prob_models/' + cond_prob_dict_filename, 'r') as cond_prob_file:
        context_dict = json.load(cond_prob_file)
    with open('cond_prob_models/' + cum_prob_dict_filename, 'r') as cum_prob_file:
        cumulative_dict = json.load(cum_prob_file)
    with open('text_files/' + message_filename, 'r', encoding='utf-8-sig') as file:
        original_message = file.read()

    print('Length of original string: ', len(original_message))

    with open('encoded_messages/' + message_filename + '_zipped.cz', 'w') as zipped_file:
        zipped_message = bytes(bits2bytes(encode(original_message, context_dict, cumulative_dict, context_chars)))
        print('Length of compressed string: ', len(zipped_message))
        print(f'Compression rate: {8.0 * len(zipped_message)/ len(original_message)} bits/byte')
        zipped_file.write(str(zipped_message))

    decompressed_message = decode(bytes2bits(zipped_message), context_dict, cumulative_dict, len(original_message), context_chars)
    print(decompressed_message)

    return 0


if __name__ == '__main__':
    main()
