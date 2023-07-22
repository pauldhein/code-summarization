import tokenize as tok
from io import BytesIO
import inspect
import pkgutil
import random
import pickle
import sys
import os
import scipy

from nltk.tokenize import word_tokenize
from tqdm import tqdm


class CodeCrawler():
    def __init__(self, corpus_name="code-comment-corpus", modules=[]):
        self.name = corpus_name
        self.mods = modules
        self.functions = dict()
        self.code_comment_pairs = dict()

    def build_function_dict(self, outpath):
        def find_functions_from_object(obj, expanded):
            results = list()
            components = inspect.getmembers(obj)
            for _, comp in components:
                if inspect.isfunction(comp) or inspect.ismethod(comp):
                    results.append(comp)
                    # results.extend(find_functions_from_callable(comp))
                elif inspect.ismodule(comp) and obj.__name__ in comp.__name__:
                    if comp.__name__ not in expanded:
                        expanded.append(comp.__name__)
                        results.extend(find_functions_from_object(comp, expanded))
                elif inspect.isclass(comp):     # and obj.__name__ in comp.__name__:
                    if comp.__name__ not in expanded:
                        expanded.append(comp.__name__)
                        results.extend(find_functions_from_class(comp, expanded))
            return results

        def find_functions_from_module(mod, expanded):
            """Returns a list of all functions from cur_module"""
            results = list()
            components = inspect.getmembers(mod)
            for _, comp in tqdm(components, desc="Searching {}".format(mod.__name__)):
                if inspect.isfunction(comp) or inspect.ismethod(comp):
                    results.append(comp)
                    # results.extend(find_functions_from_callable(comp))
                elif inspect.ismodule(comp) and mod.__name__ in comp.__name__:
                    expanded.append(comp.__name__)
                    results.extend(find_functions_from_object(comp, expanded))
                elif inspect.isclass(comp):     # and cur_module.__name__ in comp.__name__:
                    expanded.append(comp.__name__)
                    results.extend(find_functions_from_class(comp, expanded))
            return results

        def find_functions_from_class(cls, expanded):
            """Returns a list of all functions from cur_module"""
            results = list()
            try:
                components = inspect.getmembers(cls)
                for _, comp in components:
                    if inspect.isfunction(comp) or inspect.ismethod(comp):
                        results.append(comp)
                        # results.extend(find_functions_from_callable(comp))
            except ModuleNotFoundError as e:
                print(e, file=sys.stderr)
            return results

        def find_functions_from_callable(callable):
            """Returns a list of all functions from cur_module"""
            results = list()

            components = inspect.getmembers(callable)
            for _, comp in components:
                if inspect.isfunction(comp) or inspect.ismethod(comp):
                    results.append(comp)

            return results

        for mod in self.mods:
            funcs = find_functions_from_module(mod, [])
            for func in tqdm(funcs, desc="Sourcing {}".format(mod.__name__)):
                if not func.__module__ == "builtins":
                    try:
                        (_, line_num) = inspect.getsourcelines(func)
                        code = inspect.getsource(func)
                        self.functions[(func.__module__, line_num)] = code
                    except Exception as e:
                        print("Failed to get {} from {}: {}".format(func.__name__, func.__module__, e), file=sys.stderr)

        pickle_path = os.path.join(outpath, "code.pkl".format(self.name))
        pickle.dump(self.functions, open(pickle_path, "wb"))

    def build_code_comment_pairs(self, outpath):
        if not self.functions:
            raise RuntimeWarning("Function dataset has not been built!!")

        self.code_comment_pairs = dict()
        num_docs = 0
        for idx, (identifier, code) in enumerate(tqdm(self.functions.items())):
            found_doc = False
            clean_code, clean_doc = list(), ""
            token_code = list(tok.tokenize(BytesIO(code.encode('utf-8')).readline))
            for tok_type, token, (line, _), _, full_line in token_code:
                if tok_type == tok.COMMENT or tok_type == tok.ENCODING:
                    continue

                if tok_type == tok.STRING and ("\"\"\"" in token or "'''" in token):
                    full_line = full_line.strip()
                    if full_line.endswith("'''") or full_line.endswith("\"\"\""):
                        for tok_type2, token2, (line2, _), _, full_line2 in token_code:
                            if line2 == line - 1 and "def" in full_line2:
                                found_doc = True
                                break
                            elif line2 >= line:
                                break

                        if found_doc:
                            clean_token = token.strip("\"\"\"").strip("'''").strip()
                            if "\n" in clean_token:
                                nl_idx = clean_token.index("\n")
                                clean_doc += clean_token[:nl_idx]
                            else:
                                clean_doc += clean_token
                            num_docs += 1
                    else:
                        clean_code.extend(token)
                elif tok_type == tok.NEWLINE or tok_type == tok.NL:
                    clean_code.append("<NEWLINE>")
                elif tok_type == tok.INDENT:
                    clean_code.append("<TAB>")
                elif tok_type == tok.DEDENT:
                    clean_code.append("<UNTAB>")
                elif tok_type == tok.ENDMARKER:
                    clean_code.append("<END>")
                elif tok_type == tok.NUMBER:
                    clean_code.append("<NUMBER>")
                elif tok_type == tok.STRING:
                    clean_code.append("<STRING>")
                elif tok_type == tok.NAME:
                    clean_code.append(token)
                else:
                    clean_code.extend(token.split())

            if found_doc:
                clean_doc = word_tokenize(clean_doc)
                clean_code = ["<BoC>"] + clean_code + ["<EoC>"]
                clean_doc = ["<BoL>"] + clean_doc + ["<EoL>"]
                self.code_comment_pairs[identifier] = (clean_code, clean_doc)

        pickle_path = os.path.join(outpath, "{}.pkl".format(self.name))
        pickle.dump(self.code_comment_pairs, open(pickle_path, "wb"))

    def get_sentence_output(self, code_file_path, comm_file_path):
        if not self.code_comment_pairs:
            raise RuntimeWarning("Code/comment dataset has not been built!!")

        outcode = open(code_file_path, "w")
        outcomm = open(comm_file_path, "w")

        for code, comment in self.code_comment_pairs.values():
            outcode.write("{}\n".format(" ".join(code)))
            outcomm.write("{}\n".format(" ".join(comment)))

        outcode.close()
        outcomm.close()

    def init_functions_from_file(self, filepath):
        self.functions = pickle.load(open(filepath), "rb")

    def init_code_comment_corpus_from_file(self, filepath):
        self.code_comment_pairs = pickle.load(open(filepath), "rb")

    def split_code_comment_data(self, train_size=0.8, val_size=0.15):
        if not self.code_comment_pairs:
            raise RuntimeWarning("Code/comment dataset has not been built!!")

        arr_data = list(self.code_comment_pairs.values())
        random.shuffle(arr_data)
        total_length = len(arr_data)
        train_length = int(train_size * total_length)
        val_length = int(val_size * total_length) + train_length

        (train_code, train_comm) = map(list, zip(*arr_data[:train_length]))
        (val_code, val_comm) = map(list, zip(*arr_data[train_length: val_length]))
        (test_code, test_comm) = map(list, zip(*arr_data[val_length:]))

        with open("train.code", "w") as outfile:
            for line in train_code:
                outfile.write("{}\n".format(" ".join(line)))

        with open("train.nl", "w") as outfile:
            for line in train_comm:
                outfile.write("{}\n".format(" ".join(line)))

        with open("dev.code", "w") as outfile:
            for line in val_code:
                outfile.write("{}\n".format(" ".join(line)))

        with open("dev.nl", "w") as outfile:
            for line in val_comm:
                outfile.write("{}\n".format(" ".join(line)))

        with open("test.code", "w") as outfile:
            for line in test_code:
                outfile.write("{}\n".format(" ".join(line)))

        with open("test.nl", "w") as outfile:
            for line in test_comm:
                outfile.write("{}\n".format(" ".join(line)))
