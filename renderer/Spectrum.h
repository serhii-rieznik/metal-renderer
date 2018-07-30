#pragma once

#define SPECTRUM_SAMPLES    3
#define FOR_EACH_WAVELENGTH for (unsigned int i = 0; i < SPECTRUM_SAMPLES; ++i)

struct Spectrum
{
    float values[SPECTRUM_SAMPLES];
    
    float operator [](const unsigned int i) const  { return values[i]; }
    
#if (__METAL_VERSION__)
    
    // access by .values

#else
    
    float& operator [](const unsigned int i) { return values[i]; }
    
#endif
};

#if (__METAL_VERSION__)

void spectrum_set(thread Spectrum& a, float value) {
    FOR_EACH_WAVELENGTH{ a.values[i] = value; }
}

void spectrum_set(device Spectrum& a, float value) {
    FOR_EACH_WAVELENGTH{ a.values[i] = value; }
}

void spectrum_add_inplace(device Spectrum& a, thread const Spectrum& b) {
    FOR_EACH_WAVELENGTH{ a.values[i] += b.values[i]; }
}

void spectrum_add_inplace(device Spectrum& a, device const Spectrum& b) {
    FOR_EACH_WAVELENGTH{ a.values[i] += b.values[i]; }
}

void spectrum_mul_inplace(device Spectrum& a, thread const Spectrum& b) {
    FOR_EACH_WAVELENGTH{ a.values[i] *= b.values[i]; }
}

void spectrum_mul_inplace(device Spectrum& a, device const Spectrum& b) {
    FOR_EACH_WAVELENGTH{ a.values[i] *= b.values[i]; }
}

Spectrum spectrum_set(float value) {
    Spectrum a;
    spectrum_set(a, value);
    return a;
}

Spectrum spectrum_mul(thread const Spectrum& a, float b) {
    Spectrum result;
    FOR_EACH_WAVELENGTH{ result.values[i] = a.values[i] * b; }
    return result;
}

Spectrum spectrum_mul(device const Spectrum& a, float b) {
    Spectrum result;
    FOR_EACH_WAVELENGTH{ result.values[i] = a.values[i] * b; }
    return result;
}

Spectrum spectrum_mul(device const Spectrum& a, device const Spectrum& b) {
    Spectrum result;
    FOR_EACH_WAVELENGTH{ result.values[i] = a.values[i] * b.values[i]; }
    return result;
}

Spectrum spectrum_mul(thread const Spectrum& a, device const Spectrum& b) {
    Spectrum result;
    FOR_EACH_WAVELENGTH{ result.values[i] = a.values[i] * b.values[i]; }
    return result;
}

Spectrum spectrum_mul(device const Spectrum& a, thread const Spectrum& b) {
    Spectrum result;
    FOR_EACH_WAVELENGTH{ result.values[i] = a.values[i] * b.values[i]; }
    return result;
}

Spectrum spectrum_mul(thread const Spectrum& a, thread const Spectrum& b) {
    Spectrum result;
    FOR_EACH_WAVELENGTH{ result.values[i] = a.values[i] * b.values[i]; }
    return result;
}

#endif
