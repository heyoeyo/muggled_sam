#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import gzip
import lzma
import unicodedata

import torch
import torch.nn as nn

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class TextTokenizer(nn.Module):
    """
    Modified implementation of the 'SimpleTokenizer' from:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    This entire purpose of this module is to take in a string (e.g. text prompt)
    and convert it to a list of integers. It does this in 3 steps:
        1. Convert (aribtrary) unicode text input into a sequence of bytes (roughly ASCII characters)
        2. Break each input into components according to a known vocabulary
        3. Map each vocab component to an integer index

    Most of the complexity comes from step 2, which makes use of something
    called 'byte-pair encoding' (BPE), which recursively fragments of text
    into known vocabularly entries.

    The original implementation can be found here:
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/tokenizer_ve.py#L128C7-L128C22
    """

    # .................................................................................................................

    def __init__(
        self,
        vocab_size: int = 49408,
        max_context_length: int = 32,
        start_of_text_delimiter: str = "<start_of_text>",
        end_of_text_delimiter: str = "<end_of_text>",
    ):
        # Inherit from parent (only using nn.Module for storing device info!)
        super().__init__()
        self.register_buffer("device_info", torch.empty(1), persistent=False)

        # Store target vocab size, so we can truncate un-used BPE table info
        self._vocab_size = vocab_size
        self._max_context_length = max_context_length

        # Store byte (i.e. uint8) to character look-up
        self._uint8_to_str_lut = make_uint8_to_character_lut()

        # Store special delimiters
        self._start_of_text_delimiter = start_of_text_delimiter
        self._end_of_text_delimiter = end_of_text_delimiter
        self._end_of_word_delimiter = "</w>"

        # Store vocab look-ups (these get filled in after loading BPE table)
        # (called 'encoder' and 'bpe_ranks' in original code, which also included a reversed lut called 'decoder')
        self._vocab_to_idx_lut: dict[str, int] = {}  # Entries like: {"this</w>": 589, "that</w>": 682, ...}
        self._txt_pair_to_rowidx_lut: dict[tuple[str, str], int] = {}  # Entries like: {("th", "is</w>"): 77, ...}

        # Set up cache for vocab encoding results
        self._vocab_encoding_cache: dict[str, list[str]] = {}

    # .................................................................................................................

    def load_bpe_vocab(
        self,
        bpe_vocab_path: str,
        start_offset_index: int = 1,
        end_index: int = 48895,
        end_of_word_delimiter: str = "</w>",
    ):
        """
        Load BPE table. In the original SAM3 codebase, this is a file called:
            bpe_simple_vocab_16e6.txt.gz
        See: https://github.com/facebookresearch/sam3/tree/main/sam3/assets

        It's expected to contain rows of character pairings. The first few lines of the file look like:
        ╔═════════════════════════════════════════╗
        ║ "bpe_simple_vocab_16e6.txt#version: 0.2 ║
        ║ i n                                     ║
        ║ t h                                     ║
        ║ a n                                     ║
        ║ r e                                     ║
        ║ ...                                     ║
        ╚═════════════════════════════════════════╝

        If the provided 'path' input has more than 1000 characters, it will
        be assumed to be the raw BPE table text itself. This is meant to provide
        a way of directly loading the data, rather than using the file system.

        Returns:
            self
        """

        # Load BPE mapping and break into list of [a, b] pairing strings
        if bpe_vocab_path.endswith(".gz"):
            # Handle gzip (format used in original SAM3 implementation)
            with gzip.open(bpe_vocab_path, "rb") as infile:
                raw_read_str = infile.read()
        elif bpe_vocab_path.endswith(".lzma"):
            # Handle lzma (format used within this implementation)
            with lzma.open(bpe_vocab_path, "rb") as infile:
                raw_read_str = infile.read()
        elif bpe_vocab_path.endswith(".txt"):
            # Assume we've been given the uncompressed text file
            with open(bpe_vocab_path, "rb") as infile:
                raw_read_str = infile.read()
        elif len(bpe_vocab_path) > 1000:
            # Assume we've been given the vocab text directly
            raw_read_str = bpe_vocab_path
        else:
            raise TypeError("Unable to interpret BPE vocab path (expecting .gz, .lzma or .txt file...")
        raw_lines_list = raw_read_str.decode("utf-8").splitlines()

        # Only take the first 'N' entries needed to get to the target vocab size
        # (the file is ~5x bigger than what we actually need to use)
        num_single_char_entries = len(self._uint8_to_str_lut) * 2
        num_start_end_entries = 2
        num_bpe_entries = self._vocab_size - (num_single_char_entries + num_start_end_entries)
        raw_lines_list = raw_lines_list[start_offset_index : (start_offset_index + num_bpe_entries)]
        str_pairs_list = tuple(tuple(line.split()) for line in raw_lines_list)

        # Build up vocabulary (order is important, must match how the model was trained!)
        # - First 256 entries are all known single-character values
        # - Next 256 are all single-character values with 'end-of-word' delimiter
        # - Next N (48894 by default) are the (joined) BPE pairs from loaded file
        # - Last 2 are the start-of/end-of-text entries
        vocab = list(self._uint8_to_str_lut.values())
        vocab.extend([f"{v}{end_of_word_delimiter}" for v in self._uint8_to_str_lut.values()])
        for str_pair in str_pairs_list:
            vocab.append("".join(str_pair))
        vocab.append(self._start_of_text_delimiter)
        vocab.append(self._end_of_text_delimiter)

        # Build vocab-to-index lookups
        vocab_size = len(vocab)
        vocab_idx_seq = range(vocab_size)
        self._vocab_to_idx_lut = dict(zip(vocab, vocab_idx_seq))
        if vocab_size != self._vocab_size:
            raise Warning(f"Vocabulary size ({vocab_size}) is different than expected ({self._vocab_size})...")

        # Create mapping of string pairs to their corresponding row-index in the BPE file
        self._txt_pair_to_rowidx_lut = dict(zip(str_pairs_list, range(len(str_pairs_list))))

        # Store start/end indexes for re-use (we'll need this often)
        self._sot_vocab_idx = self._vocab_to_idx_lut[self._start_of_text_delimiter]
        self._eot_vocab_idx = self._vocab_to_idx_lut[self._end_of_text_delimiter]
        self._end_of_word_delimiter = end_of_word_delimiter

        # Reset cache
        self._vocab_encoding_cache = {t: [t] for t in [self._start_of_text_delimiter, self._end_of_text_delimiter]}

        return self

    # .................................................................................................................

    def text_to_vocab_index(
        self, text: str, limit_context_length: bool = False, raise_length_error: bool = True
    ) -> Tensor:
        """
        Takes in text and outputs a tensor where each entry is an index
        corresponding to an entry in a (learned) vocabulary.

        This is (roughly) the 'encode' function from the original implementation:
        https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/tokenizer_ve.py#L206

        Returns:
            vocab_index_tensor_bn
            -> Shape: BxN, where B is 1 (including for batching) and N is number of tokens
            -> Each token is a integer (int64) corresponding to it's position in a learned vocabulary
        """

        # Sanity check
        assert len(self._vocab_to_idx_lut) > 0, "BPE has not been loaded!, use .load_bpe_vocab(...)"

        cleaned_text = text.lower().strip()
        vocab_idxs_list = [self._sot_vocab_idx]
        for txt_group in split_to_text_groups(cleaned_text):
            # Remap the text into a small (256) set of unique characters
            # -> For typical ASCII text this won't do anything
            # -> For non-ASCII characters, this will expand to gibberish-looking text
            #    For example: ['Déjà', 'Vu'] maps to ['DÃ©jÃł', 'Vu']
            remapped_txt_group = "".join(self._uint8_to_str_lut[char_uint8] for char_uint8 in txt_group.encode("utf-8"))

            # Break text into known vocab entries and map to corresponding integer index
            vocab_pieces_list = self._convert_to_vocab_components(remapped_txt_group)
            vocab_idxs_list.extend(self._vocab_to_idx_lut[vocab_txt] for vocab_txt in vocab_pieces_list)
        vocab_idxs_list.append(self._eot_vocab_idx)

        # Error if output is too large
        result_length = len(vocab_idxs_list)
        if limit_context_length and result_length > self._max_context_length:
            if raise_length_error:
                raise ValueError(f"Encoding exceeds length limit! ({result_length} vs. {self._max_context_length})")

            # Truncate to max length and re-add end-of-text indicator
            vocab_idxs_list = vocab_idxs_list[0 : (self._max_context_length - 1)]
            vocab_idxs_list.append(self._eot_vocab_idx)

        # Convert to tensor for output
        return torch.tensor(vocab_idxs_list, dtype=torch.int64, device=self.device_info.device).unsqueeze(0)

    # .................................................................................................................

    def _convert_to_vocab_components(self, text: str) -> list[str]:
        """
        Function for encoding a piece of text into a known vocabulary.
        Works by first splitting the text into pairs of characters,
        and then merging pairs according to a given BPE table.
        Merged results are referred to as 'text pieces' in this function.

        This is then repeated, recursively, so that the merged
        text pieces can themselves be merged with other characters/pieces.

        Merging ends when all text has been merged, or if none of the
        pairings exist in the BPE table.

        Example, using the input: 'visual'
          This is first broken into 'pieces' with an end delimiter: [v, i, s, u, a, l</w>]
          Then the function repeatedly merges pairs of pieces, according to the BPE table.
          Pairs are merge in order of first appearance in the table:

              [v, i, s, u, a, l</w>]  (starting sequence)
              [v, i, s, u, al</w>]    (merge 'a' and 'l</w>',     row 56 of table)
              [v, is, u, al</w>]      (merge 'i' and 's',         row 85 of table)
              [vis, u, al</w>]        (merge 'v' and 'is',        row 791 of table)
              [vis, ual</w>]          (merge 'u' and 'al</w>',    row 1411 of table)
              [visual</w>]            (merge 'vis' and 'ual</w>', row 6685 of table)

        This is process is handled by the 'bpe' function in the original implementation:
        https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/tokenizer_ve.py#L168

        Returns:
            final_vocab_list
        """

        # Use cached results whenever possible
        if text in self._vocab_encoding_cache:
            return self._vocab_encoding_cache[text]

        # Sanity check. Handle empty inputs & single characters
        num_chars = len(text)
        assert num_chars > 0, "Must provide non-empty text!"
        if num_chars == 1:
            return [f"{text}</w>"]

        # Use individual character as initial 'pieces', with delimiter on last character
        # -> Example for input: 'visual' -> (v,i,s,u,a,l</w>)
        text_pieces = tuple([*f"{text[:-1]}", f"{text[-1]}{self._end_of_word_delimiter}"])

        # Combine text pieces according to BPE table until there is only 1 piece left or no pieces in table
        out_of_bounds_idx = len(self._txt_pair_to_rowidx_lut)
        get_rowidx_of_txt_pair = lambda p: self._txt_pair_to_rowidx_lut.get(p, out_of_bounds_idx)
        while len(text_pieces) > 1:

            # Get all unique pairs of text pieces. Each piece can be 1 or more characters
            # -> Example for pieces: (v,i,s,u,a,l</w>) -> {(v,i), (i,s), (s,u), (u,a), (a,l</w>)}
            unique_consecutive_pairs = set(zip(text_pieces[:-1], text_pieces[1:]))

            # Get text pair that appears 'highest' in the stored bpe vocab table
            pair_to_merge = min(unique_consecutive_pairs, key=get_rowidx_of_txt_pair)
            if pair_to_merge not in self._txt_pair_to_rowidx_lut:
                break

            # For clarity
            merged_pair = "".join(pair_to_merge)
            txt_piece_1, txt_piece_2 = pair_to_merge

            # Loop over text pieces and merge target text pair
            # e.g. If target pair is: [a,b] and pieces are: [a,a,b,b,c,c,a,b]
            #                               then result is: [a,ab ,b,c,c,ab ]
            # (Note that the outer loop just repeats this process)
            new_pieces_list = []
            remaining_pieces = text_pieces[0:]
            while len(remaining_pieces) > 0:

                # Special check: We may not find piece 1 in the text!
                # -> This shouldn't happen on the first loop iteration, but can happen on later
                #    iterations, if we've already found the piece earlier and no duplicates exist
                # -> In these cases, we're done merging pieces, so bail
                if not txt_piece_1 in remaining_pieces:
                    new_pieces_list.extend(remaining_pieces)
                    break

                # Find the first piece check that the next piece is the matching pair piece
                # -> If not, we take the pieces as-is and continue searching
                # -> This could happen if searching for pair: (i,a) in word: 'initial' for example
                #    we would search for 'i' and find (i,n), continue to (i,t) and finally find: (i,a)
                pidx = remaining_pieces.index(txt_piece_1)
                next_piece_idx = 1 + pidx
                if remaining_pieces[next_piece_idx] == txt_piece_2:

                    # Take all pieces up to the first piece as well as the merged pair
                    # -> Example, if pieces: [a,b,c,d,e,f,g] and pair=(c,d), then take: [a,b,cd]
                    new_pieces_list.extend(remaining_pieces[:pidx])
                    new_pieces_list.append(merged_pair)

                    # Remove the pieces we found and continue searching
                    # -> Example, now have [a,b,cd], continue search through remaining pieces: [e,f,g]
                    remaining_pieces = remaining_pieces[(2 + pidx) :]

                    continue

                # Here, we haven't found the merge pair, so just take all pieces
                # up to this point as-is and continue searching the left-over pieces
                new_pieces_list.extend(remaining_pieces[:next_piece_idx])
                remaining_pieces = remaining_pieces[next_piece_idx:]

            # Now repeat merging with new (smaller) set of pieces
            text_pieces = tuple(new_pieces_list)

        # Cache result for re-use
        self._vocab_encoding_cache[text] = text_pieces

        return text_pieces

    # .................................................................................................................

    def _debug_index_to_text(self, text_index_list: Tensor | list[int]) -> list[str]:
        """
        Helper used to reverse a vocabulary (index) encoding. Useful for
        checking that the indexing (roughly) matches the text that
        was originally encoded.
        Returns:
            list_of_original_strings
        """

        try:
            # Force tensors/numpy arrays to list of integers
            text_index_list = text_index_list.tolist()
        except AttributeError:
            # Happens if we're given a list/tuple. We don't need to do anything
            pass

        reversed_idx_to_vocab_lut = {v: k for k, v in self._vocab_to_idx_lut.items()}
        return [reversed_idx_to_vocab_lut[int(token)] for token in text_index_list]

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Helper functions


def make_uint8_to_character_lut() -> dict[int, str]:
    """
    This function creates a lookup-table that maps every number
    between 0 and 255 (inclusive) to a single (string) character.

    For common ASCII characters, the index is just the ASCII code
    (e.g. index 50 maps to '2', index 90 maps to 'Z').
    See: https://www.ascii-code.com/

    However, the 'control characters' (indexes 0-to-32) and
    some uncommon codes (127-to-160 and 173), are mapped to
    characters outside of the ASCII range, by using the
    'Latin Extended-A' unicode block (e.g. indexes >255)
    See: https://en.wikipedia.org/wiki/Latin_Extended-A

    This is kind of a strange mapping, because it has the
    effect of mapping uncommon ASCII codes (single bytes)
    into pairs of bytes.

    The original implementation calls this 'bytes_to_unicode':
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/tokenizer_ve.py#L30

    Returns:
        uint8_code_to_character_lut

        -> character_code_to_string_lut is a dictionary with integer keys ranging
           from 0-to-255 and values which are unicode characters (as strings)
        -> The strings themselves are all individual unicode characters,
           with codes ranging from 33 to 323
    """

    # For clarity. These are the ranges of ASCII codes we'll store 'directly'
    # (Note: range(33, 127) means '33 up to and including 126')
    direct_mapping_ranges = tuple([range(33, 127), range(161, 173), range(174, 256)])

    offset_idx = 256
    uint8_to_char_lut = {}
    for idx in range(256):
        is_direct_map = any(idx in direct_range for direct_range in direct_mapping_ranges)
        value = idx
        if not is_direct_map:
            value = offset_idx
            offset_idx += 1
        uint8_to_char_lut[idx] = chr(value)

    return uint8_to_char_lut


# .....................................................................................................................


def split_to_text_groups(text: str):
    """
    Helper function used to break text into smaller groups.
    This implementation is a *very* verbose way to do this,
    but avoids the need for an external 'regex' library.

    It splits text according to the following rules:
        - Splits certain contractions: "don't you'll fall" -> [don, 't, you, 'll, fall]
        - Splits whole words: "A sunny day" -> [A, sunny, day]
        - Splits numbers into separate digits: "The 5525th index" -> [The, 5, 5, 2, 5, th, index]
        - Splits blocks of punctuation: "Wait... what? Where?!?!" -> [Wait, ..., what, ?, Where, ?!?!]

    The original code does this with a single regex, see:
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/tokenizer_ve.py#L157-L159

    However, this requires an external regex library so we're avoiding it here
    (the built-in python regex doesn't handle non-english characters properly...?)
    """

    # For clarity
    target_contractions = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
    check_is_letter = lambda char: unicodedata.category(char).startswith("L")
    check_is_number = lambda char: unicodedata.category(char).startswith("N")

    # Read text character-by-character looking for groupings as we go
    num_chars = len(text)
    char_groups_list = []
    curr_idx = -1
    while True:

        # Stop once we've read all the characters
        curr_idx += 1
        if curr_idx >= num_chars:
            break
        curr_char = text[curr_idx]

        # Skip white space
        if curr_char.isspace():
            continue

        # Store contractions as their own group
        if curr_char == "'":
            is_contraction = False
            next_few_chars = text[(curr_idx) : (curr_idx + 3)]
            for targ_cont in target_contractions:
                is_contraction = next_few_chars.startswith(targ_cont)
                if is_contraction:
                    char_groups_list.append(targ_cont)
                    curr_idx = curr_idx + len(targ_cont)
                    break

            # Skip remaining parsing checks if we found a contraction
            if is_contraction:
                continue

        # Group letters to form whole-words
        if check_is_letter(curr_char):
            new_group = []
            future_text = text[curr_idx:]
            for future_char in future_text:
                if not check_is_letter(future_char):
                    break
                new_group.append(future_char)

            # Store final (combined) word
            char_groups_list.append("".join(new_group))
            curr_idx = curr_idx + len(new_group) - 1
            continue

        # Separate numbers
        if check_is_number(curr_char):
            char_groups_list.append(curr_char)
            continue

        # Group everything that isn't a letter/number (e.g. punctuation)
        new_group = []
        future_text = text[curr_idx:]
        for future_char in future_text:
            if check_is_letter(future_char):
                break
            if check_is_number(future_char):
                break
            new_group.append(future_char)

        # Store final group of characters
        char_groups_list.append("".join(new_group))
        curr_idx = curr_idx + len(new_group) - 1

    return char_groups_list
