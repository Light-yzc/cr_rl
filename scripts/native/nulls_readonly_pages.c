/* Page transport for an existing MemoryBackend. Never writes target memory.
 * Linux x86_64 static executable also runs under Android x86_64 as root.
 * stdin: ASCII page_count followed by hexadecimal addresses, one per line.
 * stdout: READY errno, then PAGE errno and a base64 page line on success.
 * Encoding avoids byte corruption by emulator adbd CRLF translation.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>

static int io_all(int fd, void *buf, size_t n, int writing) {
    unsigned char *p=buf;
    while(n) {
        ssize_t k=writing ? write(fd,p,n) : read(fd,p,n);
        if(k<0 && errno==EINTR) continue;
        if(k<=0) return -1;
        p+=k;n-=k;
    }
    return 0;
}
int main(int argc,char **argv) {
    /* Some emulator adbd builds still allocate a PTY with shell -T.
     * Disable its CRLF translation and canonical input for byte-exact transport. */
    struct termios tty;
    if(tcgetattr(0,&tty)==0) {cfmakeraw(&tty);tcsetattr(0,TCSANOW,&tty);}
    if(tcgetattr(1,&tty)==0) {cfmakeraw(&tty);tcsetattr(1,TCSANOW,&tty);}
    if(argc!=2) return 2;
    char *end;long pid=strtol(argv[1],&end,10);
    if(*end || pid<=0) return 2;
    char path[64];snprintf(path,sizeof(path),"/proc/%ld/mem",pid);
    int fd=open(path,O_RDONLY|O_CLOEXEC);
    int32_t status=fd<0?errno:0;
    char header[40];int h=snprintf(header,sizeof(header),"READY %d\n",status);
    if(io_all(1,header,h,1) || status) return 3;
    uint64_t addresses[1024];unsigned char page[4096];
    for(;;) {
        uint32_t n;
        if(scanf("%u",&n)!=1) break;
        if(n==0 || n>1024) return 4;
        for(uint32_t i=0;i<n;i++) if(scanf("%" SCNx64,&addresses[i])!=1) return 4;
        for(uint32_t i=0;i<n;i++) {
            status=0;
            if(addresses[i]<65536 || addresses[i]%4096 || addresses[i]>INT64_MAX-4096) status=EINVAL;
            else {
                ssize_t got;
                do {got=pread(fd,page,4096,(off_t)addresses[i]);} while(got<0 && errno==EINTR);
                if(got!=4096) status=got<0?errno:EIO;
            }
            h=snprintf(header,sizeof(header),"PAGE %d\n",status);
            if(io_all(1,header,h,1)) return 5;
            if(!status) {
                static const char alphabet[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                char encoded[5465];size_t nout=0;
                for(size_t k=0;k<4096;k+=3) {
                    unsigned v=(unsigned)page[k]<<16;
                    if(k+1<4096)v|=(unsigned)page[k+1]<<8;
                    if(k+2<4096)v|=page[k+2];
                    encoded[nout++]=alphabet[(v>>18)&63];encoded[nout++]=alphabet[(v>>12)&63];
                    encoded[nout++]=k+1<4096?alphabet[(v>>6)&63]:'=';
                    encoded[nout++]=k+2<4096?alphabet[v&63]:'=';
                }
                encoded[nout++]='\n';if(io_all(1,encoded,nout,1))return 5;
            }
        }
    }
    close(fd);return 0;
}
