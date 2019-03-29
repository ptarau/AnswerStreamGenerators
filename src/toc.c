#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#define BSIZE (1<<16)

int doFile(char* fname);

int main() {
  int keep=0;
  char s[BSIZE];
  while(1) {
    if(NULL==(void*)gets(s)) break;

    if(0==strncmp(s,"\\end{document}",14)) break;
      
      if(0==strncmp(s,"\\input",6)) {
          char *fname = s+7;
          //fname[strlen(fname)-1]='\0';
          //fprintf(stderr,"HERE %s\n",s);
          if(!doFile(fname)) {
              fprintf(stderr,"\n\n%s\n\n","       !!! missing include !!!");
              //fprintf(stderr,"errno: %d %s\n",errno,strerror(errno));

              exit(1);
          };
      };

    if(0==strncmp(s,"\\begin{code}",12) || 0==strncmp(s,"\\begin{codeh}",13)) {
      keep=1;
    }
    else if (0==strncmp(s,"\\end{code}",10) || 0==strncmp(s,"\\end{codeh}",11)) {
      keep=0;
      puts("");
    }
    else if(keep>0) puts(s);
  }
}

int doFile(char *fname) {
    FILE *f;
    f=fopen(fname,"r");
    fprintf(stderr,"%s\n",fname);
    if(NULL==f) return 0;
    int keep=0;

    char s[BSIZE];
    while(1) {
        if(NULL==(void*)fgets(s,BSIZE,f)) break;
        
        if(0==strncmp(s,"\\end{document}",14)) break;
        
        if(0==strncmp(s,"\\begin{code}",12) || 0==strncmp(s,"\\begin{codeh}",13)) {
            keep=1;
        }
        else if (0==strncmp(s,"\\end{code}",10) || 0==strncmp(s,"\\end{codeh}",11)) {
            keep=0;
            puts("");
        }
        else if(keep>0) puts(s);
    }
    fclose(f);
    return 1;
}