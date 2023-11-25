import ctypes
from ctypes import (
    c_bool,
    c_char,
    c_char_p,
    c_int,
    c_int8,
    c_int32,
    c_uint8,
    c_uint32,
    c_size_t,
    c_float,
    c_double,
    c_void_p,
    pointer,
    POINTER,
    _Pointer,  # type: ignore
    Structure,
    Array,
)

from typing import Union, List

# Shared library will be stored in ./lib/ as a llama.dll file, .so for unix
lib_path = "./lib/llama.dll"

# Instantiate Library Loader
lib = ctypes.CDLL(lib_path)

# struct llama_model;
llama_model_p = c_void_p

# struct llama_context;
llama_context_p = c_void_p

# typedef int32_t llama_pos;
llama_pos = c_int32

# typedef int32_t llama_token;
llama_token = c_int32
llama_token_p = POINTER(llama_token)

# typedef int32_t llama_seq_id;
llama_seq_id = c_int32

# typedef struct llama_token_data {
#     llama_token id; // token id
#     float logit;    // log-odds of the token
#     float p;        // probability of the token
# } llama_token_data;
class llama_token_data(Structure):
    _fields_ = [
        ("id", llama_token),
        ("logit", c_float),
        ("p", c_float),
    ]

llama_token_data_p = POINTER(llama_token_data)

# typedef struct llama_token_data_array {
#     llama_token_data * data;
#     size_t size;
#     bool sorted;
# } llama_token_data_array;
class llama_token_data_array(Structure):
    _fields_ = [
        ("data", llama_token_data_p),
        ("size", c_size_t),
        ("sorted", c_bool),
    ]


llama_token_data_array_p = POINTER(llama_token_data_array)

# typedef void (*llama_progress_callback)(float progress, void *ctx);
llama_progress_callback = ctypes.CFUNCTYPE(None, c_float, c_void_p)

# // Input data for llama_decode
# // A llama_batch object can contain input about one or many sequences
# // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
# //
# // - token  : the token ids of the input (used when embd is NULL)
# // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
# // - pos    : the positions of the respective token in the sequence
# // - seq_id : the sequence to which the respective token belongs
# // - logits : if zero, the logits for the respective token will not be output
# //
# typedef struct llama_batch {
#     int32_t n_tokens;

#     llama_token  *  token;
#     float        *  embd;
#     llama_pos    *  pos;
#     int32_t      *  n_seq_id;
#     llama_seq_id ** seq_id;
#     int8_t       *  logits;


#     // NOTE: helpers for smooth API transition - can be deprecated in the future
#     //       for future-proof code, use the above fields instead and ignore everything below
#     //
#     // pos[i] = all_pos_0 + i*all_pos_1
#     //
#     llama_pos    all_pos_0;  // used if pos == NULL
#     llama_pos    all_pos_1;  // used if pos == NULL
#     llama_seq_id all_seq_id; // used if seq_id == NULL
# } llama_batch;
class llama_batch(Structure):
    """Input data for llama_decode

    A llama_batch object can contain input about one or many sequences

    The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens

    Attributes:
        token (ctypes.Array[llama_token]): the token ids of the input (used when embd is NULL)
        embd (ctypes.Array[ctypes.c_float]): token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
        pos (ctypes.Array[ctypes.Array[llama_pos]]): the positions of the respective token in the sequence
        seq_id (ctypes.Array[ctypes.Array[llama_seq_id]]): the sequence to which the respective token belongs
    """

    _fields_ = [
        ("n_tokens", c_int32),
        ("token", POINTER(llama_token)),
        ("embd", POINTER(c_float)),
        ("pos", POINTER(llama_pos)),
        ("n_seq_id", POINTER(c_int32)),
        ("seq_id", POINTER(POINTER(llama_seq_id))),
        ("logits", POINTER(c_int8)),
        ("all_pos_0", llama_pos),
        ("all_pos_1", llama_pos),
        ("all_seq_id", llama_seq_id),
    ]
    


# struct llama_model_params {
#     int32_t n_gpu_layers; // number of layers to store in VRAM
#     int32_t main_gpu;     // the GPU that is used for scratch and small tensors
#     const float * tensor_split; // how to split layers across multiple GPUs (size: LLAMA_MAX_DEVICES)

#     // called with a progress value between 0 and 1, pass NULL to disable
#     llama_progress_callback progress_callback;
#     // context pointer passed to the progress callback
#     void * progress_callback_user_data;


#     // Keep the booleans together to avoid misalignment during copy-by-value.
#     bool vocab_only; // only load the vocabulary, no weights
#     bool use_mmap;   // use mmap if possible
#     bool use_mlock;  // force system to keep model in RAM
# };
class llama_model_params(Structure):
    _fields_ = [
        ("n_gpu_layers", c_int32),
        ("main_gpu", c_int32),
        ("tensor_split", POINTER(c_float)),
        ("progress_callback", llama_progress_callback),
        ("progress_callback_user_data", c_void_p),
        ("vocab_only", c_bool),
        ("use_mmap", c_bool),
        ("use_mlock", c_bool),
    ]


# struct llama_context_params {
#     uint32_t seed;              // RNG seed, -1 for random
#     uint32_t n_ctx;             // text context, 0 = from model
#     uint32_t n_batch;           // prompt processing maximum batch size
#     uint32_t n_threads;         // number of threads to use for generation
#     uint32_t n_threads_batch;   // number of threads to use for batch processing
#     int8_t   rope_scaling_type; // RoPE scaling type, from `enum llama_rope_scaling_type`

#     // ref: https://github.com/ggerganov/llama.cpp/pull/2054
#     float    rope_freq_base;   // RoPE base frequency, 0 = from model
#     float    rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
#     float    yarn_ext_factor;  // YaRN extrapolation mix factor, NaN = from model
#     float    yarn_attn_factor; // YaRN magnitude scaling factor
#     float    yarn_beta_fast;   // YaRN low correction dim
#     float    yarn_beta_slow;   // YaRN high correction dim
#     uint32_t yarn_orig_ctx;    // YaRN original context size


#     // Keep the booleans together to avoid misalignment during copy-by-value.
#     bool mul_mat_q;  // if true, use experimental mul_mat_q kernels (DEPRECATED - always true)
#     bool f16_kv;     // use fp16 for KV cache, fp32 otherwise
#     bool logits_all; // the llama_eval() call computes all logits, not just the last one
#     bool embedding;  // embedding mode only
# };
class llama_context_params(Structure):
    _fields_ = [
        ("seed", c_uint32),
        ("n_ctx", c_uint32),
        ("n_batch", c_uint32),
        ("n_threads", c_uint32),
        ("n_threads_batch", c_uint32),
        ("rope_scaling_type", c_int8),
        ("rope_freq_base", c_float),
        ("rope_freq_scale", c_float),
        ("yarn_ext_factor", c_float),
        ("yarn_attn_factor", c_float),
        ("yarn_beta_fast", c_float),
        ("yarn_beta_slow", c_float),
        ("yarn_orig_ctx", c_uint32),
        ("mul_mat_q", c_bool),
        ("f16_kv", c_bool),
        ("logits_all", c_bool),
        ("embedding", c_bool),
    ]
    


# Default parameter helpers
# LLAMA_API struct llama_model_params llama_model_default_params(void);
def llama_model_default_params() -> llama_model_params:
    return lib.llama_model_default_params()

lib.llama_model_default_params.argtypes = []
lib.llama_model_default_params.restype = llama_model_params

# LLAMA_API struct llama_context_params llama_context_default_params(void);
def llama_context_default_params() -> llama_context_params:
    return lib.llama_context_default_params()

lib.llama_context_default_params.argtypes = []
lib.llama_context_default_params.restype = llama_context_params


# LLAMA_API struct llama_model * llama_load_model_from_file(
#                          const char * path_model,
#         struct llama_model_params     params);
def llama_load_model_from_file(
    path_model: bytes, params: llama_model_params
) -> llama_model_p:
    return lib.llama_load_model_from_file(path_model, params)

lib.llama_load_model_from_file.argtypes = [c_char_p, llama_model_params]
lib.llama_load_model_from_file.restype = llama_model_p

# LLAMA_API struct llama_context * llama_new_context_with_model(
#                  struct llama_model * model,
#         struct llama_context_params   params);
def llama_new_context_with_model(
    model: llama_model_p, params: llama_context_params
) -> llama_context_p:
    return lib.llama_new_context_with_model(model, params)


lib.llama_new_context_with_model.argtypes = [llama_model_p, llama_context_params]
lib.llama_new_context_with_model.restype = llama_context_p

# // Frees all allocated memory
# LLAMA_API void llama_free(struct llama_context * ctx);
def llama_free(ctx: llama_context_p):
    """Frees all allocated memory"""
    return lib.llama_free(ctx)


lib.llama_free.argtypes = [llama_context_p]
lib.llama_free.restype = None

# // Call once at the end of the program - currently only used for MPI
# LLAMA_API void llama_backend_free(void);
def llama_backend_free():
    """Call once at the end of the program - currently only used for MPI"""
    return lib.llama_backend_free()


lib.llama_backend_free.argtypes = []
lib.llama_backend_free.restype = None

# LLAMA_API void llama_free_model(struct llama_model * model);
def llama_free_model(model: llama_model_p):
    return lib.llama_free_model(model)


lib.llama_free_model.argtypes = [llama_model_p]
lib.llama_free_model.restype = None

# LLAMA_API int llama_n_vocab    (const struct llama_model * model);
def llama_n_vocab(model: llama_model_p) -> int:
    return lib.llama_n_vocab(model)


lib.llama_n_vocab.argtypes = [llama_model_p]
lib.llama_n_vocab.restype = c_int

# LLAMA_API int llama_n_ctx      (const struct llama_context * ctx);
def llama_n_ctx(ctx: llama_context_p) -> int:
    return lib.llama_n_ctx(ctx)

lib.llama_n_ctx.argtypes = [llama_context_p]
lib.llama_n_ctx.restype = c_int

# LLAMA_API int llama_n_ctx_train(const struct llama_model * model);
def llama_n_ctx_train(model: llama_model_p) -> int:
    return lib.llama_n_ctx_train(model)

lib.llama_n_ctx_train.argtypes = [llama_model_p]
lib.llama_n_ctx_train.restype = c_int

# // Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
# // Each token can be assigned up to n_seq_max sequence ids
# // The batch has to be freed with llama_batch_free()
# // If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
# // Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
# // The rest of the llama_batch members are allocated with size n_tokens
# // All members are left uninitialized
# LLAMA_API struct llama_batch llama_batch_init(
#         int32_t n_tokens,
#         int32_t embd,
#         int32_t n_seq_max);
def llama_batch_init(
    n_tokens: Union[c_int32, int],
    embd: Union[c_int32, int],
    n_seq_max: Union[c_int32, int],
) -> llama_batch:
    """Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    Each token can be assigned up to n_seq_max sequence ids
    The batch has to be freed with llama_batch_free()
    If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
    The rest of the llama_batch members are allocated with size n_tokens
    All members are left uninitialized"""
    return lib.llama_batch_init(n_tokens, embd, n_seq_max)


lib.llama_batch_init.argtypes = [c_int32, c_int32, c_int32]
lib.llama_batch_init.restype = llama_batch

# // Frees a batch of tokens allocated with llama_batch_init()
# LLAMA_API void llama_batch_free(struct llama_batch batch);
def llama_batch_free(batch: llama_batch):
    """Frees a batch of tokens allocated with llama_batch_init()"""
    return lib.llama_batch_free(batch)

lib.llama_batch_free.argtypes = [llama_batch]
lib.llama_batch_free.restype = None

# // Positive return values does not mean a fatal error, but rather a warning.
# //   0 - success
# //   1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
# // < 0 - error
# LLAMA_API int llama_decode(
#         struct llama_context * ctx,
#           struct llama_batch   batch);
def llama_decode(ctx: llama_context_p, batch: llama_batch) -> int:
    """Positive return values does not mean a fatal error, but rather a warning.
    0 - success
    1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    < 0 - error"""
    return lib.llama_decode(ctx, batch)

lib.llama_decode.argtypes = [llama_context_p, llama_batch]
lib.llama_decode.restype = c_int

# // Token logits obtained from the last call to llama_eval()
# // The logits for the last token are stored in the last row
# // Logits for which llama_batch.logits[i] == 0 are undefined
# // Rows: n_tokens provided with llama_batch
# // Cols: n_vocab
# LLAMA_API float * llama_get_logits(struct llama_context * ctx);
def llama_get_logits(
    ctx: llama_context_p,
):  # type: (...) -> Array[float] # type: ignore
    """Token logits obtained from the last call to llama_eval()
    The logits for the last token are stored in the last row
    Logits for which llama_batch.logits[i] == 0 are undefined
    Rows: n_tokens provided with llama_batch
    Cols: n_vocab"""
    return lib.llama_get_logits(ctx)


lib.llama_get_logits.argtypes = [llama_context_p]
lib.llama_get_logits.restype = POINTER(c_float)

# // Logits for the ith token. Equivalent to:
# // llama_get_logits(ctx) + i*n_vocab
# LLAMA_API float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);
def llama_get_logits_ith(
    ctx: llama_context_p, i: Union[c_int32, int]
):  # type: (...) -> Array[float] # type: ignore
    """Logits for the ith token. Equivalent to:
    llama_get_logits(ctx) + i*n_vocab"""
    return lib.llama_get_logits_ith(ctx, i)


lib.llama_get_logits_ith.argtypes = [llama_context_p, c_int32]
lib.llama_get_logits_ith.restype = POINTER(c_float)

# // Special tokens

# LLAMA_API llama_token llama_token_bos(const struct llama_model * model); // beginning-of-sentence
def llama_token_bos(model: llama_model_p) -> int:
    """beginning-of-sentence"""
    return lib.llama_token_bos(model)


lib.llama_token_bos.argtypes = [llama_model_p]
lib.llama_token_bos.restype = llama_token


# LLAMA_API llama_token llama_token_eos(const struct llama_model * model); // end-of-sentence
def llama_token_eos(model: llama_model_p) -> int:
    """end-of-sentence"""
    return lib.llama_token_eos(model)

lib.llama_token_eos.argtypes = [llama_model_p]
lib.llama_token_eos.restype = llama_token

# LLAMA_API llama_token llama_token_nl (const struct llama_model * model); // next-line
def llama_token_nl(model: llama_model_p) -> int:
    """next-line"""
    return lib.llama_token_nl(model)

lib.llama_token_nl.argtypes = [llama_model_p]
lib.llama_token_nl.restype = llama_token

# //
# // Tokenization
# //


# /// @details Convert the provided text into tokens.
# /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
# /// @return Returns the number of tokens on success, no more than n_max_tokens
# /// @return Returns a negative number on failure - the number of tokens that would have been returned
# /// @param special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated as plaintext.
# ///                Does not insert a leading space.
# LLAMA_API int llama_tokenize(
#     const struct llama_model * model,
#                   const char * text,
#                          int   text_len,
#                  llama_token * tokens,
#                          int   n_max_tokens,
#                         bool   add_bos,
#                         bool   special);
def llama_tokenize(
    model: llama_model_p,
    text: bytes,
    text_len: Union[c_int, int],
    tokens,  # type: Array[llama_token]
    n_max_tokens: Union[c_int, int],
    add_bos: Union[c_bool, bool],
    special: Union[c_bool, bool],
) -> int:
    """Convert the provided text into tokens."""
    return lib.llama_tokenize(
        model, text, text_len, tokens, n_max_tokens, add_bos, special
    )

lib.llama_tokenize.argtypes = [
    llama_model_p,
    c_char_p,
    c_int,
    llama_token_p,
    c_int,
    c_bool,
    c_bool,
]
lib.llama_tokenize.restype = c_int

def llama_token_to_piece(
    model: llama_model_p,
    token: Union[llama_token, int],
    buf: Union[c_char_p, bytes],
    length: Union[c_int, int],
) -> int:
    """Token Id -> Piece.
    Uses the vocabulary in the provided context.
    Does not write null terminator to the buffer.
    User code is responsible to remove the leading whitespace of the first non-BOS token when decoding multiple tokens.
    """
    return lib.llama_token_to_piece(model, token, buf, length)

lib.llama_token_to_piece.argtypes = [llama_model_p, llama_token, c_char_p, c_int]
lib.llama_token_to_piece.restype = c_int

# /// @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
# /// @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
# LLAMA_API void llama_sample_repetition_penalties(
#         struct llama_context * ctx,
#       llama_token_data_array * candidates,
#            const llama_token * last_tokens,
#                       size_t   penalty_last_n,
#                        float   penalty_repeat,
#                        float   penalty_freq,
#                        float   penalty_present);
def llama_sample_repetition_penalties(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    last_tokens_data,  # type: Array[llama_token]
    penalty_last_n: Union[c_size_t, int],
    penalty_repeat: Union[c_float, float],
    penalty_freq: Union[c_float, float],
    penalty_present: Union[c_float, float],
):
    """Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
    Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
    """
    return lib.llama_sample_repetition_penalties(
        ctx,
        candidates,
        last_tokens_data,
        penalty_last_n,
        penalty_repeat,
        penalty_freq,
        penalty_present,
    )


lib.llama_sample_repetition_penalties.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    llama_token_p,
    c_size_t,
    c_float,
    c_float,
    c_float,
]
lib.llama_sample_repetition_penalties.restype = None

# /// @details Selects the token with the highest probability.
# ///          Does not compute the token probabilities. Use llama_sample_softmax() instead.
# LLAMA_API llama_token llama_sample_token_greedy(
#         struct llama_context * ctx,
#       llama_token_data_array * candidates);
def llama_sample_token_greedy(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
) -> int:
    return lib.llama_sample_token_greedy(ctx, candidates)

lib.llama_sample_token_greedy.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
]
lib.llama_sample_token_greedy.restype = llama_token

# Implementation of llama_batch_add in common.h
"""
void llama_batch_add(
             struct llama_batch & batch,
                    llama_token   id,
                      llama_pos   pos,
const std::vector<llama_seq_id> & seq_ids,
                           bool   logits){
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}
"""
def llama_batch_add(batch: llama_batch, token_id: int, pos: int, seq_ids: List[int], logits: bool):
    batch.token[batch.n_tokens] = token_id
    batch.pos[batch.n_tokens] = pos
    batch.n_seq_id[batch.n_tokens] = len(seq_ids)
    
    for i in range(len(seq_ids)):
        batch.seq_id[batch.n_tokens][i] = seq_ids[i]

    batch.logits[batch.n_tokens] = logits
    
    batch.n_tokens += 1
    
# Implementation of llama_batch_clear in common.h
"""
void llama_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}
""" 
def llama_batch_clear(batch: llama_batch):
    batch.n_tokens = 0