#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#define SIGN_MASK (1u << 31)

int main(void)
{
    const int nfft = 256;
    const int nbits = 10;
    const int nman = 12;
    const int nexp = 6;

    uint32_t H[256];
    uint32_t packed[512];
    int32_t output[512];
    int8_t He[256];

    uint32_t iq_mask = (1u << (nman - 1)) - 1u;
    uint32_t e_mask = (1u << nexp) - 1u;
    uint32_t real_sign_mask = 1u << (nexp + 2 * nman - 1);
    uint32_t imag_sign_mask = real_sign_mask >> nman;

    int e_p = 1 << (nexp - 1);
    int maxbit = -e_p;

    for (int i = 0; i < nfft; i++) {
        if (scanf("%" SCNu32, &H[i]) != 1) {
            fprintf(stderr, "Missing input word %d\n", i);
            return 1;
        }
    }

    for (int i = 0; i < nfft; i++) {
        int vi = (H[i] >> (nexp + nman)) & iq_mask;
        int vq = (H[i] >> nexp) & iq_mask;

        int e = H[i] & e_mask;

        if (e >= e_p)
            e -= (e_p << 1);

        He[i] = (int8_t)e;

        uint32_t x = (uint32_t)vi | (uint32_t)vq;

        if (x) {
            uint32_t m = 0xffff0000u;
            uint32_t b = 0xffffu;
            int s = 16;

            while (s > 0) {
                if (x & m) {
                    e += s;
                    x >>= s;
                }

                s >>= 1;
                m = (m >> s) & b;
                b >>= s;
            }

            if (e > maxbit)
                maxbit = e;
        }

        packed[i * 2] =
            (uint32_t)vi |
            ((H[i] & real_sign_mask) ? SIGN_MASK : 0u);

        packed[i * 2 + 1] =
            (uint32_t)vq |
            ((H[i] & imag_sign_mask) ? SIGN_MASK : 0u);
    }

    int shift = nbits - maxbit;

    for (int i = 0; i < 512; i++) {
        int e = He[i >> 1] + shift;

        int sign = 1;
        uint32_t value = packed[i];

        if (value & SIGN_MASK) {
            sign = -1;
            value &= ~SIGN_MASK;
        }

        if (e < -nman) {
            value = 0;
        } else if (e < 0) {
            value >>= -e;
        } else {
            value <<= e;
        }

        int64_t result = (int64_t)sign * value;
        output[i] = (int32_t)result;
    }

    for (int i = 0; i < 512; i++)
        printf("%" PRId32 "\n", output[i]);

    return 0;
}
