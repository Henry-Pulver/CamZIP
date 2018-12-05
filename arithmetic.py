from math import floor, ceil
from sys import stdout as so
from bisect import bisect
from typing import Dict

def make_cumulative_dict(char_probs: Dict[str, float]):
    char_probs = dict([(a, char_probs[a]) for a in char_probs if char_probs[a] > 0])  # eliminate zero probabilities
    cum_prob = [0.0]

    # compute the running sum
    for counter, character in enumerate(char_probs):  # for every probability in p
        cum_prob.append(cum_prob[counter] + char_probs[character])

    # the resulting cumulative has one too many element at the end, the sum of all probabilities
    cum_prob.pop()

    return dict([(a, mf) for a, mf in zip(char_probs, cum_prob)])


def encode_rescale(lo, hi, compressed_message, one, straddle):
    """Now we need to re-scale the interval if its end-points have bits in common,
    and output the corresponding bits where appropriate. We will do this with an
    infinite loop, that will break when none of the conditions for output / straddle
    are fulfilled"""
    quarter = int(ceil(one / 4))
    half = 2 * quarter
    threequarters = 3 * quarter
    while True:
        if hi < half:  # if lo < hi < 1/2
            # stretch the interval by 2 and output a 0 followed by 'straddle' ones (if any)
            # and zero the straddle after that. In fact, HOLD OFF on doing the stretching:
            # we will do the stretching at the end of the if statement
            compressed_message.append(0)  # append a zero to the output list y
            compressed_message += [1 for s in range(straddle)]  # extend by a sequence of 'straddle' ones
            straddle = 0  # zero the straddle counter

        elif lo >= half:  # if hi > lo >= 1/2
            # stretch the interval by 2 and substract 1, and output a 1 followed by 'straddle'
            # zeros (if any) and zero straddle after that. Again, HOLD OFF on doing the stretching
            # as this will be done after the if statement, but note that 2*interval - 1 is equivalent
            # to 2*(interval - 1/2), so for now just substract 1/2 from the interval upper and lower
            # bound (and don't forget that when we say "1/2" we mean the integer "half" we defined
            # above: this is an integer arithmetic implementation!
            compressed_message.append(1)  # append a 1 to the output list y
            compressed_message += [0 for s in range(straddle)]  # extend 'straddle' zeros
            straddle = 0  # reset the straddle counter
            lo -= half
            hi -= half  # subtract half from lo and hi

        elif lo >= quarter and hi < threequarters:  # if 1/4 < lo < hi < 3/4
            # we can increment the straddle counter and stretch the interval around
            # the half way point. This can be impemented again as 2*(interval - 1/4),
            # and as we will stretch by 2 after the if statement all that needs doing
            # for now is to subtract 1/4 from the upper and lower bound
            straddle += 1  # increment straddle
            lo -= quarter
            hi -= quarter  # subtract 'quarter' from lo and hi

        else:
            return lo, hi, compressed_message, straddle  # we break the infinite loop if the interval has reached an un-stretchable state

        lo *= 2
        hi = hi * 2 + 1  # adding 1 seems to solve a minor precision problem


def encode(input_message: str, char_probs: Dict[str, float]):
    """"Encodes input_message using probabilities char_probs"""
    precision = 32
    one = int(2**precision - 1)
    quarter = int(ceil(one/4))

    cum_prob_dict = make_cumulative_dict(char_probs)
    
    compressed_message, lo, hi, straddle = [], 0, one, 0

    for k, a in enumerate(input_message): # for every symbol

        # arithmetic coding is slower than vl_encode, so we display a "progress bar"
        # to let the user know that we are processing the file and haven't crashed...
        if k % 100 == 0:
            so.write('Arithmetic encoded %d%%    \r' % int(floor(k / len(input_message) * 100)))
            so.flush()

        lohi_range = hi - lo + 1  # The added 1 is necessary to avoid rounding issues

        # 2) narrow the interval end-points [lo,hi) to the new range [f,f+p]
        # within the old interval [lo,hi], being careful to round 'innwards'
        lo += ceil(cum_prob_dict[a] * lohi_range)
        hi = lo + floor(char_probs[a] * lohi_range)

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


def decode_rescale(lo, hi, encoded_message, position, one, value):
    quarter = int(ceil(one / 4))
    half = 2 * quarter
    threequarters = 3 * quarter
    while True:
        if hi < half:
            # do nothing
            pass
        elif lo >= half:
            lo = lo - half
            hi = hi - half
            value = value - half
        elif lo >= quarter and hi < threequarters:
            lo = lo - quarter
            hi = hi - quarter
            value = value - quarter
        else:
            return lo, hi, position, value
        lo = 2 * lo
        hi = 2 * hi + 1
        value = 2 * value + encoded_message[position]
        position += 1
        if position == len(encoded_message):
            raise NameError('Unable to decompress')


def decode(encoded_message: str, char_probs: Dict[str, float], num_chars: int):
    precision = 32
    one = int(2**precision - 1)

    char_probs = dict([(a, char_probs[a]) for a in char_probs if char_probs[a] > 0])
    
    alphabet = list(char_probs)
    cum_prob = [0]
    for a in char_probs:
        cum_prob.append(cum_prob[-1] + char_probs[a])
    cum_prob.pop()

    char_probs = list(char_probs.values())

    encoded_message.extend(precision * [0]) # dummy zeros to prevent index out of bound errors
    input_message = num_chars*[0] # initialise all zeros

    # initialise by taking first 'precision' bits from y and converting to a number
    value = int(''.join(str(a) for a in encoded_message[0:precision]), 2)
    position = precision # position where currently reading y
    lo, hi = 0, one

    for k in range(num_chars):
        if k % 100 == 0:
            so.write('Arithmetic decoded %d%%    \r' % int(floor(k/num_chars*100)))
            so.flush()
        print('lo: ', lo)
        print('hi: ', hi)

        lohi_range = hi - lo + 1
        print('lohi range: ', lohi_range)
        # This is an essential subtelty: the slowest part of the decoder is figuring out
        # which symbol lands us in an interval that contains the encoded binary string.
        # This can be extremely wasteful (o(n) where n is the alphabet size) if you proceed
        # by simple looping and comparing. Here we use Python's "bisect" function that
        # implements a binary search and is 100 times more efficient. Try
        # for a = [a for a in f if f[a]<(value-lo)/lohi_range)][-1] for a MUCH slower solution.
        a = bisect(cum_prob, (value-lo)/lohi_range) - 1
        input_message[k] = alphabet[a]
        print('value: ', alphabet[a])
        print()

        lo = lo + int(ceil(cum_prob[a]*lohi_range))
        hi = lo + int(floor(char_probs[a] * lohi_range))
        if lo == hi:
            raise NameError('Zero interval!')

        lo, hi, position, value = decode_rescale(lo, hi, encoded_message, position, one, value)

    return input_message
