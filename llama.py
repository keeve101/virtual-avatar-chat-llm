import llama_cpp as lib
from typing import List, Generator

from _utils import suppress_stdout_stderr

class Llama:
    def __init__(self, model_filepath: str):
        # Instantiate model
        self.model_params = lib.llama_model_default_params()
        
        with suppress_stdout_stderr(disable=False):
            self.model = lib.llama_load_model_from_file(bytes(model_filepath), self.model_params)
            
        # Instantiate context
        self.ctx_params = lib.llama_context_default_params()
        
        # Context length // ctx_params has its own n_ctx, should probably set both to the same values
        self.n_ctx = lib.llama_n_ctx_train(self.model)

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False):
        tokens = (lib.llama_token * self.n_ctx)()
        n_tokens = lib.llama_tokenize(self.model, text, len(text), tokens, self.n_ctx, add_bos, special)
        
        return tokens[:n_tokens]

    # Pseudo wrapper for llama_decode
    def decode(self, batch: lib.llama_batch):
        if (lib.llama_decode(self.ctx, batch) != 0):
            print("Problem occured calling llama_decode()")
            return 1
    
    def detokenize(self, tokens: List[int]) -> bytes:
        output = b""
        size = 32
        buf = (lib.c_char * size)()
        for token in tokens:
            n = lib.llama_token_to_piece(self.model, lib.llama_token(token), buf, size)
            output += bytes(buf[:n])

        return output[1:] if len(tokens) > 0 and tokens[0] == lib.llama_token_bos(self.model) else output
    
    # Adaptation of simple.cpp in llama.cpp/examples
    # Yields Generator object
    def generate(self, prompt: str) -> Generator[int, None, None]:
        tokens = self.tokenize(bytes(prompt, encoding="utf-8"))
        batch = lib.llama_batch_init(n_tokens=512, embd=0, n_seq_max=1)
        
        with suppress_stdout_stderr(disable=False):
            self.ctx = lib.llama_new_context_with_model(self.model, self.ctx_params)

        for i in range(len(tokens)):
            lib.llama_batch_add(batch, tokens[i], i, [0], False)
    
        # We only consider the last logit
        batch.logits[batch.n_tokens - 1] = True
    
        # Evaluate the initial prompt
        self.decode(batch)

        n_cur = batch.n_tokens
        n_decode = 0
        n_len = 512 # max seq length including prompt
        
        while (n_cur <= n_len):
            n_vocab = lib.llama_n_vocab(self.model)
            logits = lib.llama_get_logits_ith(self.ctx, batch.n_tokens - 1)
            
            candidates = (lib.llama_token_data * n_vocab)(*[lib.llama_token_data(token_id, logits[token_id], 0.0) for token_id in range(n_vocab)])
            
            candidates_p = lib.pointer(lib.llama_token_data_array(candidates, len(candidates), False))
            
            # Greedy sample the most likely token
            sampled_token_id = lib.llama_sample_token_greedy(self.ctx, candidates_p)
            
            # Continue sampling until token EOS reached or max seq length reached
            if (sampled_token_id == lib.llama_token_eos(self.model) or n_cur == n_len):
                print("\n")
                break
            
            yield sampled_token_id
            
            # Prepare the next batch, reset batch 
            lib.llama_batch_clear(batch)
            lib.llama_batch_add(batch, sampled_token_id, n_cur, [0], True)
            
            n_decode += 1
            n_cur += 1

            # Evaluate batch
            self.decode(batch)
            
        # Free the batch after every generate call
        lib.llama_batch_free(batch)