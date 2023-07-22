

all_orig_symbols = list()
with open("code-sentences.output", "r") as infile:
    for line in infile.readlines():
        all_orig_symbols.extend(line.strip().split(" "))
unique_orig_symbols = set(all_orig_symbols)
total_syms = len(all_orig_symbols)

orig_symbol_counts = dict()
for sym in all_orig_symbols:
    if sym in orig_symbol_counts:
        orig_symbol_counts[sym] += 1
    else:
        orig_symbol_counts[sym] = 1

all_vector_symbols = list()
with open("code-vectors.txt", "r") as infile:
    for idx, line in enumerate(infile.readlines()):
        if idx == 0:
            continue
        cells = line.strip().split(" ")
        all_vector_symbols.append(cells[0])
all_vector_symbols = set(all_vector_symbols)

in_both = list(unique_orig_symbols.intersection(all_vector_symbols))
in_orig_not_vecs = list(unique_orig_symbols - all_vector_symbols)
in_vecs_not_orig = list(all_vector_symbols - unique_orig_symbols)

in_both_counts = [(sym, orig_symbol_counts[sym]) for sym in in_both]
in_both_counts.sort(key=lambda tup: tup[1], reverse=True)
in_both_stats = ["{}\t\t{}\t\t{:.8f}".format(sym, count, count / total_syms) for sym, count in in_both_counts]

in_orig_not_vecs_counts = [(sym, orig_symbol_counts[sym]) for sym in in_orig_not_vecs]
in_orig_not_vecs_counts.sort(key=lambda tup: tup[1], reverse=True)
in_orig_not_vecs_stats = ["{}\t\t{}\t\t{:.8f}".format(sym, count, count / total_syms) for sym, count in in_orig_not_vecs_counts]

with open("in_both.txt", "w") as outfile:
    outfile.write("{}\n".format("\n".join(in_both)))

with open("in_orig_not_vecs.txt", "w") as outfile:
    outfile.write("{}\n".format("\n".join(in_orig_not_vecs)))

with open("in_vecs_not_orig.txt", "w") as outfile:
    outfile.write("{}\n".format("\n".join(in_vecs_not_orig)))

with open("in_both_stats.txt", "w") as outfile:
    outfile.write("{}\n".format("\n".join(in_both_stats)))

with open("in_orig_not_vecs_stats.txt", "w") as outfile:
    outfile.write("{}\n".format("\n".join(in_orig_not_vecs_stats)))
