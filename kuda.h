#ifndef KUDA_H
#define KUDA_H

typedef struct{
    unsigned width;
    unsigned height;
    unsigned char *image; //RGBA
    
}rgb_image;

void takeimagevalue(const char* filename, rgb_image *img);
void saveimagegray(const char* filename, rgb_image *img);
void transformToGrayCuda(rgb_image *img);
__global__ void setPixelToGrayscale(unsigned char *image, unsigned width, unsigned height);

#endif ;