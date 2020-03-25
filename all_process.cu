   // #include "../common/common.h"
   #include <cuda_runtime.h>
   #include <cuda.h>
   #include <stdio.h>
   #include <stdlib.h>
   #include <time.h>
   //#include "io.h"
   #include "kuda.h"
   #include "lodepng.h"
   
   /*
    * This example helps to visualize the relationship between thread/block IDs and
    * offsets into data. For each CUDA thread, this example displays the
    * intra-block thread ID, the inter-block block ID, the global coordinate of a
    * thread, the calculated offset into input data, and the input data at that
    * offset.
    // */
   
void printMatrixGambar(int *C, const int nx, const int ny)
{
    int *ic = C;
    //printf("\nMatrix Gambar: (%d.%d)\n", nx, ny);
    FILE * fp = NULL;
    fp = fopen("matrixgambar.txt", "w+");

    if(fp == NULL){
        printf("Error creating results file\n");
        exit(1);
    }
    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            //printf("%3d", ic[ix]);
            fprintf(fp, " %d   ", ic[ix]);
   
        }
        fprintf(fp, "\n");
        ic += nx;
        //printf("\n");
    }
    printf("matrix gambar disimpan di matrixgambar.txt\n");
    printf("\n");
    fclose(fp);
    return;
}
   
void printMatrixGlcm(int *C, const int max)
{
    int *ic = C;
    //printf("\nMatrix GLCM: (%d.%d)\n", max+1, max+1);
    FILE * fp = NULL;
    fp = fopen("matrixglcm.txt", "w+");

    if(fp == NULL){
        printf("Error creating results file\n");
        exit(1);
    }
    for (int iy = 0; iy <= max; iy++)
    {
        for (int ix = 0; ix <= max; ix++)
        {
            //printf("%3d  ", ic[ix]);
            fprintf(fp, "%d  ", ic[ix]);
   
        }
        fprintf(fp, "\n\n");
        ic += (max+1);
        //printf("\n");
    }
   
    printf("\n");
    fclose(fp);
    return;
}

void printMatrixNormalization(float *C, const int nx, const int ny)
{
    float *ic = C;
    //printf("\nMatrix Normalisasi GLCM : (%d.%d)\n", nx, ny);
   
    FILE * fp = NULL;
    fp = fopen("matrixglcmnormalisasi.txt", "w+");

    if(fp == NULL){
        printf("Error creating results file\n");
        exit(1);
    }

    
    
    

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
           // printf("%f ", ic[ix]);
            fprintf(fp, "%.7f  ", ic[ix]);
   
        }
        fprintf(fp, "\n\n");
        ic += nx;
        //printf("\n");
    }
    fclose(fp);
    printf("\n");
}


__global__ void glcm_calculation(int *A,int *glcm,float *glcmNorm, const int nx, const int ny,int maxx)
{

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
 
    
    
    //unsigned int idr = iy * (maxx+1) + ix;
    
    
    int k,l;
    int p;
   

    //Calculate GLCM
    if(idx < nx*ny ){
         for(k=0;k<=maxx;k++){
             for(l=0;l<=maxx;l++){
                if((A[idx]==k) && (A[idx+1]==l)){
                    p=((maxx+1)*k) +l;
                    glcm[p]+=1;
                 }
             }
         }
    }


    //Normalization
    int sum;
    sum = 0;
    if(idx<(maxx+1)*(maxx+1)){
         for(k=0;k<((maxx+1)*(maxx+1));k++){
             sum+=glcm[k];
         }
     }
    // if(ix<1){
    //     printf("sum %d \n ",sum);
    // }
    if(idx<((maxx+1)*(maxx+1))){
            glcmNorm[idx] = float(glcm[idx])/float(sum);
    }
     
    float sums;

    if(ix<1){
        for(k=0;k<((maxx+1)*(maxx+1));k++){
            sums += glcmNorm[k];
            
        }
    }
       
    float f1;

    f1=0;
    if(ix<1){
        for(k=0;k<((maxx+1)*(maxx+1));k++){
            f1 = f1 + glcmNorm[k];
            
        }
    }
    //mat[offset] = sqrt(mat[offset]);        
    
    float f2 = 0;
    if(ix<1){
        for(k=0;k<((maxx+1)*(maxx+1));k++){
            f2 = f2 + k*k*sums;
            
        }
    }
    
    float f3;
    f3 = sqrt(f1);

    
    float f4;

    if(ix<1){
        for(k=0;k<((maxx+1)*(maxx+1));k++){
            f4 += (glcmNorm[k] * log10f(glcmNorm[k]));
            
        }
    }

    //float sum_average=0;


    // float f5;
    // if(ix<1){
    //     for(k=0;k<((maxx+1)*(maxx+1));k++){
    //         f2 = f2 + k*k*sums;
            
    //     }
    // }
    
    // for (int j = 0, int i = 0; j<DIM, i<DIM; j++,i++){
    //         for (int k = DIM*j; k<DIM*(j+1); k++)
    //         f5 += i*mat[k];
    //     } 
    
    // float f6;
    // for (int i = 0; i<DIM; i++ ){
    //     mat2[offset]= (i-f5)*(i-f5)*mat[offset];
    //     for (int j=0; j<DIM; j++){
    //         f6 += mat2[row*DIM*j];
    //     }
    // }
    // if(row<DIM){
    //         printf("array di device %d : %f \n",offset,mat[tidx]);
    //         //printf("array di device %d : %f \n",offset,mat2[tidx]);
    //         //mat[offset]=mat[offset]/sum;
    // }
    if(ix<1){
        printf("ASM : %.1f\n", f1);
        printf("Contrast : %.1f\n",f2);
        printf("Energy : %.1f\n",f3);
        printf("Entropy : %.1f\n",f4);
        //printf("Miu : %.1f\n",f5);
        //printf("Variance : %.1f\n",f6);    
    }
   
}
void takeimagevalue(const char* filename, rgb_image *img)
{
   
     unsigned error;
     unsigned char* png;
     size_t pngsize;;
   
     lodepng_load_file(&png, &pngsize, filename);
     error = lodepng_decode32(&img->image, &img->width, &img->height, png, pngsize);
     //int i,j;
    //  FILE * fs = NULL;
    //  fs = fopen("matrixgambar.txt", "w+");
    //  if(fs == NULL){
    //          printf("Error creating results file\n");
    //          exit(1);
    //  }
   
     //int max =0;
   
   
    //  for(i=0;i<img->width;i++){
    //          for(j=0;j<img->height;j++){
    //                fprintf(fs, "%d ",img->image[i]);
    //                if(img->image[i]>max){
    //                        max = img->image[i];
    //                }
    //          }
    //          fprintf(fs,"\n");
    //  }
    //  printf("max : %d\n",max);
    //  printf("nilai pixel di simpan di matrixgambarrgb.txt\n");
     if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
	//clock_t timer_diff2 = clock() - timer_start;
	
	//printf("waktu salin CUDA-RAM: %gs\n", (timer_diff2 / (double)CLOCKS_PER_SEC));
	
	
	
}

void transformToGrayCuda(rgb_image *img){   
	unsigned char* image = img->image;
    unsigned char* image_d;
    unsigned width = img->width;
    unsigned height = img->height;
    int N = (int)width * (int)height; 
    size_t size = N * 4 * sizeof(unsigned char);
	

	int device_count = 0;
	cudaError_t status = cudaGetDeviceCount(&device_count);
	
	status = cudaMalloc((void **) &image_d, size);
	

	cudaMemcpy(image_d, image,  size, cudaMemcpyHostToDevice);
		
	dim3 block_size(16, 16);
	dim3 num_blocks(img->width / block_size.x, img->height / block_size.y);
    setPixelToGrayscale<<<num_blocks, block_size>>>(image_d, img->width, img->height);
    

	
	cudaMemcpy(image, image_d, size, cudaMemcpyDeviceToHost);
	
	cudaFree(image_d);
}

__global__
void setPixelToGrayscale(unsigned char *image, unsigned width, unsigned height)
{
    float gray;
    float r, g, b;
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		r = image[4 * width * y + 4 * x + 0];
		g = image[4 * width * y + 4 * x + 1];
		b = image[4 * width * y + 4 * x + 2];
		gray =.299f*r + .587f*g + .114f*b;
		image[4 * width * y + 4 * x + 0] = gray;
		image[4 * width * y + 4 * x + 1] = gray;
		image[4 * width * y + 4 * x + 2] = gray;
		image[4 * width * y + 4 * x + 3] = 255;
	}
	
}

void saveimagegray(const char* filename, rgb_image *img)
{
  /*Encode the image*/
  unsigned error = lodepng_encode32_file(filename, img->image, img->width, img->height);

  /*if there's an error, display it*/
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
}

   
int main(int argc, char *argv[])
{
    printf("%s Starting...\n", argv[0]);

    const char* filename = argc > 1 ? argv[1] : "test.png";
    rgb_image img;
    takeimagevalue(filename, &img);
    transformToGrayCuda(&img);
    //printf(" di main %d %d\n",img.width,img.height);
    int nx =img.width;
    int ny =img.height;
    //int nxy = nx * ny;
    int nBytes =  img.width * img.height * sizeof(float);
    int *h_A;
    int max=0;
    h_A = (int *)malloc(nBytes);
    // set matrix dimension
    // iniitialize host matrix with image pixel
    for(int i =0;i<img.width*img.height;i++){
         h_A[i]=img.image[i];
         if(h_A[i]>max){
             //find graylevel image
             max=h_A[i];
         }
    }
    printf("max:%d \n",max);
    saveimagegray("hasil_gray.png", &img);
    // get device information
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    
    //GLCM

    // malloc host memory glcm
    int kBytes = (max+1) * (max+1) * sizeof(float);
    int *h_glcm;
    h_glcm = (int *)malloc(kBytes);
 
    

    // iniitialize host matrix glcm with integer
    for (int i = 0; i < (max+1)*(max+1); i++)
    {
        h_glcm[i] = 0;
    }
    //Normalization
    // malloc host memory glcm Normalization
    
    float *h_glcmNorm;
    h_glcmNorm = (float *)malloc(kBytes);
 
    

    // iniitialize host matrix glcm Normalization with integer
    for (int i = 0; i < (max+1)*(max+1); i++)
    {
        h_glcmNorm[i] = 0.0;
    }

    //Penampung nilai glcm di Host
    int *gpuRef;
    gpuRef = (int *)malloc(kBytes);
    memset(gpuRef, 0, kBytes);
    
    printMatrixGambar(h_A, img.width, img.height);
    // malloc device memory glcm
    int *d_glcm;
    cudaMalloc((void **)&d_glcm, kBytes);
    cudaGetLastError();

    // malloc device memory glcm Normalization
    float *d_glcmNorm;
    cudaMalloc((void **)&d_glcmNorm, kBytes);
    cudaGetLastError();
    // malloc device memory
    int *d_MatA;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaGetLastError();
    // transfer data from host to device glcm
    cudaMemcpy(d_glcm, h_glcm, kBytes, cudaMemcpyHostToDevice);
    cudaGetLastError();
    // transfer data from host to device glcm Normalization
    cudaMemcpy(d_glcmNorm, h_glcmNorm, kBytes, cudaMemcpyHostToDevice);
    cudaGetLastError();

    // transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaGetLastError();

    // set up exec1
    dim3 block(ny);
    dim3 grid((nx + block.x - 1) / block.x, (nx + block.y - 1) / block.y);
    cudaGetLastError();

    //printf("ok\n");

    clock_t start, end;
     double t = 0;
     start = clock();
    // invoke the kernel
    glcm_calculation<<<ny, nx>>>(d_MatA,d_glcm,d_glcmNorm, ny, nx,max);
    cudaGetLastError();
    
    //copy from device to host
    cudaMemcpy(gpuRef, d_glcm, kBytes, cudaMemcpyDeviceToHost);

     //copy from device to host
     cudaMemcpy(h_glcmNorm, d_glcmNorm, kBytes, cudaMemcpyDeviceToHost);
     

    printMatrixGlcm(gpuRef,max);
    printMatrixNormalization(h_glcmNorm,(max+1), (max+1));
    
    printf("matrix glcm disimpan di matrixglcm.txt\n");
    printf("matrix glcm normalisasi disimpan di matrixglcmnormalisasi.tx\n");
    end = clock();
     t = ((double) (end - start))/CLOCKS_PER_SEC;
     
     printf("waktu eksekusi: %f\n",t);
    // free host and devide memory
    cudaFree(d_MatA);cudaFree(d_glcm);cudaFree(d_glcmNorm);
    free(h_A);

    // reset device
   cudaDeviceReset();

    return (0);
}
   