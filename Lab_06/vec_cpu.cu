#define N 10000

void compute( float * a, float * b, float * c )
{
        for (int i=0; i<N; i++)
                c[i] = b[i] / a[i];
}

int main(void)
{
        float a[N], b[N], c[N];

        for (int i=0; i<N; i++)
        {
                a[i] = cos(i);
                b[i] = sin(i);
        }

        compute(a, b, c);

        for (int i=0; i<10; i++)
                printf(">> for i %d, computes %f.\n", i, c[i]);



        return 0;
}
