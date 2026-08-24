/* host_runner - run robot.cact queries through the independent C99 engine.
 * Format mirrors the official engine: system+tools prefix, then query turns.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nd_cact.h"
#include "nd_grammar.h"
#include "nd_model.h"
#include "nd_sample.h"
#include "nd_tokenizer.h"

static nd_model    mdl;
static nd_grammar  gram;

#ifndef ND_BOS_ID
#define ND_BOS_ID 2
#endif

static void gen(const char *query)
{
    static uint32_t ids[1024];
    static char     suf[2048];
    nd_sampler      smp;
    const float    *lg = NULL;
    int             n;
    uint32_t        i;
    int             produced = 0;

    snprintf(suf, sizeof(suf), "\n%s<|im_end|>\n<|im_start|>assistant\n", query);

    nd_model_rewind(&mdl);
    nd_sampler_init(&smp, &mdl.tok, &gram);

    n = nd_tok_encode_ex(&mdl.tok, suf, strlen(suf), ids,
                         (uint32_t)(sizeof(ids) / sizeof(ids[0])), 0);
    if (n < 0) { printf("ERR prompt_too_long\n"); return; }

    for (i = 0; i < (uint32_t)n; i++)
        lg = nd_model_step_hidden(&mdl, ids[i]);

    printf("OUT: ");
    for (i = 0; i < 160; i++) {
        uint32_t t = nd_sample_hidden(&mdl, &smp, lg);
        lg = NULL;
        if (t == (uint32_t)-1) { printf("[no_legal]"); break; }
        if (t == ND_EOS_ID || t == ND_IM_END_ID) break;
        nd_sample_accept(&smp, t);
        uint16_t pcn16 = 0;
        const char *pc = nd_tok_piece(&mdl.tok, t, &pcn16);
        int pcn = (int)pcn16;
        if (pc && pcn > 0) fwrite(pc, 1, (size_t)pcn, stdout);
        fflush(stdout);
        produced++;
        lg = nd_model_step_hidden(&mdl, t);
    }
    printf("\n---\n");
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s model.cact schema.json [query]\n", argv[0]); return 1; }

    FILE  *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void  *blob = malloc((size_t)sz);
    if (fread(blob, 1, (size_t)sz, f) != (size_t)sz) { perror("read"); return 1; }
    fclose(f);

    f = fopen(argv[2], "rb");
    if (!f) { perror(argv[2]); return 1; }
    static char raw[65536], comp[65536];
    size_t rn = fread(raw, 1, sizeof(raw) - 1, f);
    fclose(f);
    raw[rn] = 0;
    /* strip whitespace to compact json like the device does */
    if (nd_json_compact(raw, rn, comp, sizeof(comp)) < 0) { fprintf(stderr, "compact failed\n"); return 1; }

    if (nd_model_open(&mdl, blob, (size_t)sz) != 0) { fprintf(stderr, "model_open failed\n"); return 1; }
    if (nd_grammar_compile(&gram, comp, strlen(comp), NULL) != 0) { fprintf(stderr, "grammar failed\n"); return 1; }

    /* prime prefix: <|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\n<tools>JSON</tools> */
    {
        static uint32_t ids[1024];
        static char     pre[32768];
        const char     *sysm = "device: domestic robot; locale: en-US";
        int n = snprintf(pre, sizeof(pre),
                         "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n<tools>%s</tools>",
                         sysm, comp);
        nd_model_reset(&mdl);
        ids[0] = ND_BOS_ID;
        int k = nd_tok_encode(&mdl.tok, pre, (size_t)n, ids + 1,
                              (uint32_t)(sizeof(ids) / sizeof(ids[0]) - 1));
        if (k < 0) { fprintf(stderr, "prefix too long\n"); return 1; }
        for (int i = 0; i < k + 1; i++) nd_model_step_hidden(&mdl, ids[i]);
        nd_model_snapshot(&mdl);
        fprintf(stderr, "primed %d tokens\n", k + 1);
    }

    if (argc > 3) {
        gen(argv[3]);
    } else {
        char line[4096];
        while (fgets(line, sizeof(line), stdin)) {
            size_t L = strlen(line);
            while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
            if (!L) continue;
            gen(line);
        }
    }
    return 0;
}
