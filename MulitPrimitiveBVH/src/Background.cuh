#pragma once
#include "Image.h"
#include "sutil\vec_math.h"


class Background {

public:
    cudaTextureObject_t tex;

    __host__ Background() {}
    __host__ Background(const char* filename) {
        loadBackground(filename);
    }

    __host__ ~Background() {}

    __host__ void loadBackground(const char* filename) {
        Img sky_img = loadImage(filename); // Load the image using loadImage function
        sky_img.generateCudaTexture();
        this->tex = sky_img.getTexture();
    }

};