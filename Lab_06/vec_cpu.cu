#include <stdio.h>
#include <time.h>

#define N 100000

void add( int * a, int * b, int * c )
{
        for (int i=0; i<N; i++)
                c[i] = a[i] + b[i];
}

int main(void)
{
        int a[N], b[N], c[N];

        for (int i=0; i<N; i++)
        {
                a[i] = -i;
                b[i] = i * i;
        }

        clock_t start = clock();
        add(a, b, c);
        clock_t duration = clock() - start;
        for (int i=0; i<N; i++)
                printf(">> %d + %d = %d\n", a[i], b[i], c[i]);
        printf("Duration: %ld\n", duration);

        return 0;
}
