import numpy as np
import time
import matplotlib.pyplot as plt


def hash1_remainder(i, nbins):
    '''
    modulo of i
    '''
    return i % nbins


def hash2_midsquare(i, nbins):
    '''
    modulo of middle digits of square of i
    '''
    midsquare = int(str(i * i).zfill(9)[3:6])
    return midsquare % nbins


def hash3_folding(i, nbins):
    '''
    modulo of sum of the segmented digits of i
    '''
    istring = str(i).zfill(9)
    isum = int(istring[:3]) + int(istring[3:6]) + int(istring[6:])
    return isum % nbins


def apply_hash(integers, nbins, hash_func):
    start = time.time()
    hashes = np.asarray([hash_func(i, nbins) for i in integers])
    time_taken = time.time() - start
    return hashes, time_taken


def check_vs_uniform(hashes, nbins, nint):
    '''
    Check how close hashes is to a uniform distribution
    across nbins bins
    Returns sum of absolute errors
    '''
    ref_dist = np.full(nbins, nint/float(nbins))
    hash_dist = np.histogram(hashes, nbins, density=False)[0]
    return np.sum(np.abs(ref_dist - hash_dist))


# number of integers to hash
nint = 10000
# number of hash slots (bins)
nbins = 100
# number of times to repeat test
nrepeats = 10

results = {}
labels = ['remainder', 'midsquare', 'folding']

for l in labels:
    results[l] = {}
    results[l]['timings'] = []
    results[l]['unf_deviation'] = []
    results[l]['collisions'] = []

for t in range(nrepeats):
    integers = np.random.randint(low=1, high=1000000, size=nint)
    for l,f in zip(labels,
            [hash1_remainder, hash2_midsquare, hash3_folding]):

        # create hashes and check computational speed
        hashes, time_taken = apply_hash(integers, nbins, f)
        results[l]['timings'].append(time_taken)

        # deviation from uniform distribution
        unf_deviation = check_vs_uniform(hashes, nbins, nint)
        results[l]['unf_deviation'].append(unf_deviation)

        # number of hash collisions
        collisions = len(hashes) - len(set(hashes))
        results[l]['collisions'].append(collisions)

for l in labels:
    plt.plot(range(nrepeats), results[l]['unf_deviation'], label=l)
plt.show()
