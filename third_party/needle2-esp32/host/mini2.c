#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nd_cact.h"
#include "nd_model.h"
static nd_model mdl;
int main(int argc, char **argv){
    FILE *f=fopen(argv[1],"rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    void *blob=malloc((size_t)sz); if(fread(blob,1,(size_t)sz,f)!=(size_t)sz)return 1; fclose(f);
    if(nd_model_open(&mdl,blob,(size_t)sz)!=0)return 1;
    nd_model_reset(&mdl);
    f=fopen(argv[2],"rb"); fseek(f,0,SEEK_END); long pn=ftell(f); fseek(f,0,SEEK_SET);
    char *txt=malloc((size_t)pn+1); if(fread(txt,1,(size_t)pn,f)!=(size_t)pn)return 1; fclose(f);
    static uint32_t ids[2048];
    ids[0]=ND_BOS_ID;
    int n=nd_tok_encode(&mdl.tok, txt, (size_t)pn, ids+1, 2000);
    int total=n+1;
    const float *lg=NULL;
    for(int i=0;i<total;i++) lg=nd_model_step_hidden(&mdl,ids[i]);
    printf("GEN: ");
    for(int s=0;s<60;s++){
        float tmp[8192];
        const float *fl=nd_model_logits_all(&mdl, lg);
        memcpy(tmp,fl,sizeof(tmp));
        int best=0; for(int v=1;v<8192;v++) if(tmp[v]>tmp[best]) best=v;
        if(best==ND_EOS_ID||best==ND_IM_END_ID) break;
        uint16_t plen=0; const char *pc=nd_tok_piece(&mdl.tok,best,&plen);
        if(pc&&plen>0) fwrite(pc,1,plen,stdout);
        fflush(stdout);
        lg=nd_model_step_hidden(&mdl,(uint32_t)best);
    }
    printf("\n");
    return 0;
}
