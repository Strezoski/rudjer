import matplotlib.pyplot as plt
from random import shuffle

MIN_LENGTH = 60


def read_data(dataset_path):
    """
    Reads sequences from fasta file
    :param dataset_path: path of file that contains sequences in fasta format
    :return:
    """
    seq = []
    for line in open(dataset_path):
        if line[0] == '>':
            continue
        line = line.upper().strip()
        if (len(line)-line.count('N')) < MIN_LENGTH:  # N denotes any nucleotide
            continue
        seq.append(line.upper().strip())
    return seq


def plot_dist(seq_list, bins=20, key='len'):
    """
    Plots distribution of sequence lengths or the number of G's.
    :param seq_list: list of sequences
    :param bins: number of bins for histogram
    :param key: 'len' for the distribution of lengths, 'g' for the distribution of G's
    :return:
    """
    values = []
    for cur_seq in seq_list:
        if key == 'len':
            values.append(len(cur_seq))
        elif key == 'g':
            values.append(cur_seq.count('G'))
    plt.hist(values, bins)
    if key == 'len':
        plt.xlabel('Length')
    elif key == 'g':
        plt.xlabel('# G nucleotides')
    plt.ylabel('# occurences')
    plt.show()


def equalize_length_dist(ref_array, change_array):
    """
    Modifies given change_array so that the distribution of sequence lenghts is similar to the
    distribution of sequence lengths of a ref_array.
    :param ref_array: reference array
    :param change_array: array that needs to be modified
    :return: modified change_array
    """
    ref_dist = get_length_dist(ref_array)
    change_array.sort(key=len)
    new_array = []
    position = 0
    for length in ref_dist:
        num = int(round(ref_dist[length]*len(change_array), 0))
        for i in range(position, position+num):
            best_subseq = get_g_abundant_subseq(change_array[i], length)
            if best_subseq != '':
                new_array.append(best_subseq)
        position += num
    return new_array


def get_g_abundant_subseq(sequence, length):
    """
    Chooses a sub-sequence with more G nucleotides.
    If a sequence is shorter than given length, returns whole sequence
    :param sequence:
    :param length: length of a sub-sequence
    :return: sub-sequence with more G nucleotides
    """
    if len(sequence) < length:
        return sequence
    best_seq = ''
    best_num_g = 0
    for i in range(0, (len(sequence)-length+1)):
        subseq = sequence[i:(i+length)]
        num_g = subseq.count('G')
        if num_g > best_num_g:
            best_num_g = num_g
            best_seq = subseq
    return best_seq


def get_length_dist(seq_list):
    """
    Finds distribution of sequence lengths. For each length stores the probability of appearance
    :param seq_list: list of sequences
    :return: distribution of sequence lenghts
    """
    dist = dict()
    for seq in seq_list:
        n = len(seq)
        if not dist.__contains__(n):
            dist[n] = 0
        dist[n] += 1
    for key in dist:
        dist[key] /= float(len(seq_list))
    return dist


def encode_sequence():
    """
    Encodes letters in nucleotide sequence with binary codes
    :return: sequence code
    """
    code = dict()
    code['A'] = '0,0,0,1'
    code['C'] = '0,0,1,0'
    code['G'] = '0,1,0,0'
    code['T'] = '1,0,0,0'

    return code


def write_training_data(output_path, pos_seq, neg_seq):
    """
    Writes training data in file.
    :param output_path:
    :param pos_seq: sequences that should be labelled as positive examples
    :param neg_seq: sequences that should be labelled as negative examples
    :return:
    """
    out_file = open(output_path, 'w')
    max_length = max(len(max(pos_seq, key=len)), len(max(neg_seq, key=len)))
    num_instances = len(pos_seq)+len(neg_seq)
    out_file.write('%d, %d, %d\n' % (num_instances, max_length, 4))
    code = encode_sequence()
    all_seq = [s + ';1' for s in pos_seq]+[s + ';0' for s in neg_seq]
    shuffle(all_seq)
    write_sequences(out_file, all_seq, code, max_length)


def write_sequences(out_file, instances, code, max_length):
    """
    Writes sequences in file.
    :param out_file:
    :param instances: sequences with class label
    :param code: binary code of nucleotides
    :param max_length: length of  the longest sequence
    :return:
    """
    for inst in instances:
        inst_el = inst.split(';')
        seq = inst_el[0]
        label = inst_el[1]
        padding_position = max_length-(len(seq)-seq.count('N'))  # N denotes any nucleotide
        out_file.write('%d;' % padding_position)
        for ncl in seq:
            if ncl == 'N':
                continue
            out_file.write('%s;' % code[ncl])
        out_file.write('%s\n' % label)


negative_seq = read_data('/media/maria/Windows/Documents and Settings/maria/Documents/My projects/'
                  'G4 quadruplex/data/G4_neither_PDS_nor_Kplus.fasta')
positive_seq = read_data('/media/maria/Windows/Documents and Settings/maria/Documents/My projects/'
                  'G4 quadruplex/data/G4_PDS_and_Kplus.fasta')

# testing...
# a = ['gagg', 'gggg', 'agcg','agt']
# b = ['gaggccc', 'ggggcc', 'agcgcacacacggggggcac','agt','acccgg']
# negative_seq = equalize_length_dist(a, b)

negative_seq = equalize_length_dist(positive_seq, negative_seq)
# plot_dist(negative_seq)
# plot_dist(positive_seq)
# plot_dist(negative_seq, key='g')
# plot_dist(positive_seq, key='g')
write_training_data('/home/maria/PycharmProjects/G4Quadruplex/g4_training_set.txt', positive_seq, negative_seq)
