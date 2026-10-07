// Dump llama2.c run.c's logits at every position for a fixed token sequence.
// run.c is included unmodified; TESTING hides its main(). Compiled by noise_runc.py:
//   gcc <flags> -I ~/refs/inference/llama2.c -o dump dump.c -lm
// usage: ./dump model.bin tokens.i32 out.f32
#define TESTING
#include "run.c"

int main(int argc, char** argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s model.bin tokens.i32 out.f32\n", argv[0]); return 1; }
    Transformer t;
    build_transformer(&t, argv[1]);

    FILE* tf = fopen(argv[2], "rb");
    if (!tf) { perror(argv[2]); return 1; }
    fseek(tf, 0, SEEK_END);
    int n = (int)(ftell(tf) / sizeof(int));
    rewind(tf);
    int* tokens = malloc(n * sizeof(int));
    if (fread(tokens, sizeof(int), n, tf) != (size_t)n) { fprintf(stderr, "short read\n"); return 1; }
    fclose(tf);
    if (n > t.config.seq_len) { fprintf(stderr, "%d tokens > seq_len %d\n", n, t.config.seq_len); return 1; }

    FILE* out = fopen(argv[3], "wb");
    for (int pos = 0; pos < n; pos++) {
        // one token at a time through the KV cache: decode-style, unlike PyTorch's all-at-once
        float* logits = forward(&t, tokens[pos], pos);
        fwrite(logits, sizeof(float), t.config.vocab_size, out);
    }
    fclose(out);
    free(tokens);
    free_transformer(&t);
    return 0;
}
