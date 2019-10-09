#include <stdio.h>

#define N 10000  // vector size

void compute( float * a, float * b, float * c )
{
        // compute tangent for each number
        for (int i=0; i<N; i++)
                c[i] = b[i] / a[i];
}

int main(void)
{
        // initialize vectors
        float a[N], b[N], c[N];

        // define numbers as sines and cosines
        for (int i=0; i<N; i++)
        {
                a[i] = cos(i);
                b[i] = sin(i);
        }
        
        // compute tangent
        compute(a, b, c);

        // prints first 10 results
        for (int i=0; i<10; i++)
                printf(" >> tangent of %d computes as %f.\n", i, c[i]);



        return 0;
}
