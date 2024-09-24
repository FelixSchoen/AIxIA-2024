import codecs
import pickle
from pathlib import Path
from typing import List, Union

from paul3.data.text.text_dataset import TextDataset
from paul3.utils.paul_logging import get_logger
from paul3.utils.settings import Settings

settings = Settings()
LOGGER = get_logger(__name__)


def text_process_store_dual(paths_source_files, paths_target_files, path_output, vocab_size_bpe=37000, skip_bpe=False):
    path_output.parent.mkdir(parents=True, exist_ok=True)

    sentences_source = []
    sentences_target = []

    for path_source_file, path_target_file in zip(paths_source_files, paths_target_files):
        with codecs.open(path_source_file, "r", encoding="utf-8") as f:
            sentences_source.extend(f.read().split("\n"))
        with codecs.open(path_target_file, "r", encoding="utf-8") as f:
            sentences_target.extend(f.read().split("\n"))

    assert len(sentences_source) == len(
        sentences_target), f"There are a different number of source or target sequences: {len(sentences_source)} vs {len(sentences_target)}"

    import youtokentome

    if not skip_bpe:
        with codecs.open(path_output.parent.joinpath("combined.txt"), "w", encoding="utf-8") as f:
            f.writelines(sentences_source + sentences_target)

        LOGGER.info("Training BPE model...")
        youtokentome.BPE.train(data=str(path_output.parent.joinpath("combined.txt")), vocab_size=vocab_size_bpe,
                               model=str(path_output.parent.joinpath("bpe.model")), pad_id=0, bos_id=1, eos_id=2,
                               unk_id=3)

    LOGGER.info("Encoding sentences...")
    bpe_model = youtokentome.BPE(model=str(path_output.parent.joinpath("bpe.model")))
    data_source = bpe_model.encode(sentences_source, bos=False, eos=False)
    data_target = bpe_model.encode(sentences_target, bos=False, eos=False)

    del bpe_model

    LOGGER.info("Storing dataset...")
    dict_list = []
    for sentence_pair in zip(data_source, data_target):
        dict_list.append({"source": sentence_pair[0], "target": sentence_pair[1]})

    # LOGGER.info("Writing dataset...")
    # dataset_handler = WebDataSetHandler(path_output, settings.DATA_SHUFFLE_BUFFER_SIZE)
    # dataset_handler.write_data(dict_list)


def text_process_store_single(paths_source_files, path_output):
    path_output.parent.mkdir(parents=True, exist_ok=True)

    corpus = []

    for path_source_file in paths_source_files:
        with codecs.open(path_source_file, "r", encoding="utf-8") as f:
            corpus.append(f.read())

    with codecs.open(path_output.parent.joinpath("combined.txt"), "w", encoding="utf-8") as f:
        f.write("".join(corpus))

    corpus_split = []
    for corpus_slice in corpus:
        c_i = 0
        c_w = 510
        while c_i + c_w < len(corpus_slice):
            corpus_split.append(corpus_slice[c_i:c_i + c_w])
            c_i += c_w

    LOGGER.info("Encoding sentences...")
    tokeniser = TextCharacterTokeniser()
    with open(path_output.parent.joinpath("combined.txt"), "r") as f:
        all_text = f.read()
    tokeniser.create_vocabulary(all_text)
    with open(path_output.parent.joinpath("tokeniser.pkl"), "wb") as f:
        pickle.dump(tokeniser, f)

    LOGGER.info("Storing dataset...")
    i = 0
    with TextDataset.Writer(path_output) as writer:
        for sentence in corpus_split:
            sentence = sentence.replace('\r', '\n')
            encoded = tokeniser.tokenise(sentence)
            writer.write(i, {"sentence": encoded})
            i += 1


class TextCharacterTokeniser:

    def __init__(self):
        self.vocabulary = None
        self.vocab_size = None
        self.mapper_w_t_t = None
        self.mapper_t_t_w = None

    def create_vocabulary(self, corpus: str):
        vocabulary = []
        vocabulary.extend(['\n', ' ', '!', '&', "'", ',', '-', '.', ':', ';', '?'])
        vocabulary.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])
        vocabulary.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
        vocabulary = sorted(set(vocabulary))
        vocabulary.insert(0, "(pad)")
        vocabulary.insert(1, "(start)")
        vocabulary.insert(2, "(stop)")
        vocabulary.insert(3, "(unkn)")

        mapper_w_t_t = {word: index for index, word in enumerate(vocabulary)}
        mapper_t_t_w = {index: word for index, word in enumerate(vocabulary)}

        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)
        self.mapper_w_t_t = mapper_w_t_t
        self.mapper_t_t_w = mapper_t_t_w

    def tokenise(self, sentence: str):
        return [self.mapper_w_t_t[s] if s in self.mapper_w_t_t else self.mapper_w_t_t["(unkn)"] for s in sentence]

    def detokenise(self, tokens: List[int]):
        return "".join([self.mapper_t_t_w[t] if t in self.mapper_t_t_w else "(unkn)" for t in tokens])
