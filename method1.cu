
   #include <cuda_runtime.h>
   #include <cuda.h>
   #include <stdio.h>
   #include <stdlib.h>
   #include <time.h>
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
    fp = fopen("../data/matrixgambar.txt", "w+");

    if(fp == NULL){
        printf("Error creating results file\n");
        exit(1);
    }
    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            
            fprintf(fp, " %d   ", ic[ix]);
   
        }
        fprintf(fp, "\n");
        ic += nx;
    }
    
    printf("\n");
    fclose(fp);
    return;
}
   
void printMatrixGlcm(int *C, const int max,int degree)
{
    int *ic = C;
    FILE * fp = NULL;
    if(degree==0){
        fp = fopen("matrix_glcm_0.txt", "w+");
    }
    else if(degree==90){
        fp = fopen("matrix_glcm_90.txt", "w+");
    }
    else if(degree==180){
        fp = fopen("matrix_glcm_180.txt", "w+");
    }
    else if(degree==270){
        fp = fopen("matrix_glcm_270.txt", "w+");
    }
    else if(degree==45){
        fp = fopen("matrix_glcm_45.txt", "w+");
    }
    else if(degree==135){
        fp = fopen("matrix_glcm_135.txt", "w+");
    }
    else if(degree==225){
        fp = fopen("matrix_glcm_225.txt", "w+");
    }
    else if(degree==315){
        fp = fopen("matrix_glcm_315.txt", "w+");
    }

    if(fp == NULL){
        printf("Error creating results file\n");
        exit(1);
    }
    for (int iy = 0; iy <= max; iy++)
    {
        for (int ix = 0; ix <= max; ix++)
        {
            fprintf(fp, "%d  ", ic[ix]);
   
        }
        fprintf(fp, "\n\n");
        ic += (max+1);

    }
   
    printf("\n");
    fclose(fp);
    return;
}


void printMatrixNormalization(float *C, const int max,int degree)
{
    float *ic = C;
    FILE * fp = NULL;
    if(degree==0){
        fp = fopen("matrix_normalisasi_0.txt", "w+");
    }
    else if(degree==90){
        fp = fopen("matrix_normalisasi_90.txt", "w+");
    }
    else if(degree==180){   
        fp = fopen("matrix_normalisasi_180.txt", "w+");
    }
    else if(degree==270){
        fp = fopen("matrix_normalisasi_270.txt", "w+");
    }
    else if(degree==45){
        fp = fopen("matrix_normalisasi_45.txt", "w+");
    }
    else if(degree==135){
        fp = fopen("matrix_normalisasi_135.txt", "w+");
    }
    else if(degree==225){
        fp = fopen("matrix_normalisasi_225.txt", "w+");
    }
    else if(degree==315){
        fp = fopen("matrix_normalisasi_315.txt", "w+");
    }

    if(fp == NULL){
        printf("Error creating results file\n");
        exit(1);
    }
    for (int iy = 0; iy <= max; iy++)
    {
        for (int ix = 0; ix <= max; ix++)
        {
            
            fprintf(fp, "%.7f  ", ic[ix]);

        }
        fprintf(fp, "\n\n");
        ic += (max+1);
        
    }

    printf("\n");
    fclose(fp);
    return;
}


__global__ void glcm_calculation_nol(int *A,int *glcm, const int nx, const int ny,int maxx)
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
                    atomicAdd(&glcm[p],1);
                 }
             }
         }
    }

}


__global__ void glcm_calculation_180(int *A,int *glcm, const int nx, const int ny,int max){
    //int iy = threadIdx.y + blockIdx.y* blockDim.y;
    unsigned int idx =blockIdx.x*nx+threadIdx.x;
    int i;
    int k=0;
    for(i=0;i<nx;i++){
        if(idx>=i*nx && idx<((i+1) *nx)-1){
            k=max*A[idx+1]+A[idx];
            atomicAdd(&glcm[k],1);
        }
    }
}

__global__ void glcm_calculation_270(int *A,int *glcm, const int nx, const int ny,int max){
    int ix = threadIdx.x + blockIdx.x* blockDim.x;
    int iy = threadIdx.y + blockIdx.y* blockDim.y;
    unsigned int idx =iy*nx+ix;
    int i;
    int k=0;
    for(i=0;i<nx-1;i++){
        if(idx>=i*nx && idx<((i+1) *nx)){
            k=max*A[idx]+A[idx+nx];
            atomicAdd(&glcm[k],1);           
        }
    }
    __syncthreads();
}

__global__ void glcm_calculation_90(int *A,int *glcm, const int nx, const int ny,int max){
    int ix = threadIdx.x + blockIdx.x* blockDim.x;
    int iy = threadIdx.y + blockIdx.y* blockDim.y;
    unsigned int idx =iy*nx+ix;
    int i;
    int k=0;
    for(i=0;i<nx-1;i++){
        if(idx>=i*nx && idx<((i+1) *nx)){
            k=max*A[idx+nx]+A[idx];
            atomicAdd(&glcm[k],1);          
        }
    }
    __syncthreads();
}

__global__ void glcm_calculation_45(int *A,int *glcm, const int nx, const int ny,int max){
    int ix = threadIdx.x + blockIdx.x* blockDim.x;
    int iy = threadIdx.y + blockIdx.y* blockDim.y;
    unsigned int idx =iy*nx+ix;
    int i;
    int k=0;
    for(i=1;i<nx;i++){
        if(blockIdx.x==i && idx <((i+1)*nx)-1){
            k=max*A[idx]+A[idx-(nx-1)];
            atomicAdd(&glcm[k],1);
        }
    }
    __syncthreads();
}

__global__ void glcm_calculation_135(int *A,int *glcm, const int nx, const int ny,int max){
    int ix = threadIdx.x + blockIdx.x* blockDim.x;
    int iy = threadIdx.y + blockIdx.y* blockDim.y;
    unsigned int idx =iy*nx+ix;
    int i;
    int k=0;
    for(i=0;i<nx-1;i++){
        if(blockIdx.x==i && idx >i*nx){
            k=max*A[idx]+A[idx+(nx-1)];
            atomicAdd(&glcm[k],1);
        }
    }
    __syncthreads();
}

__global__ void glcm_calculation_225(int *A,int *glcm, const int nx, const int ny,int max){
    int ix = threadIdx.x + blockIdx.x* blockDim.x;
    int iy = threadIdx.y + blockIdx.y* blockDim.y;
    unsigned int idx =iy*nx+ix;
    int i;
    int k=0;
    for(i=1;i<nx;i++){
        if(blockIdx.x==i && idx >i*nx){
            k=max*A[idx]+A[idx-(nx+1)];
            atomicAdd(&glcm[k],1);
        }
    }
    __syncthreads();
}

__global__ void glcm_calculation_315(int *A,int *glcm, const int nx, const int ny,int max){
    int ix = threadIdx.x + blockIdx.x* blockDim.x;
    int iy = threadIdx.y + blockIdx.y* blockDim.y;
    unsigned int idx =iy*nx+ix;
    int i;
    int k=0;
    for(i=0;i<nx-1;i++){
        if(blockIdx.x==i && idx <((i+1)*nx)-1){
            k=max*A[idx]+A[idx+(nx+1)];
            atomicAdd(&glcm[k],1);
        }
    }
    __syncthreads();
}

__global__ void normalization(int *glcm,float *norm,int max,int sum){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * max + ix;
    __syncthreads();
    if(idx<(max+1)*(max+1)){
        norm[idx]=float(glcm[idx])/float(sum);
    }
}
__global__ void calculate_contrast(float *norm,float *contrast,int *dif,int max,float sum,int size){
    //printf("%d\n",max);
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * max + ix;
    int tid=threadIdx.x;
    //printf("%d\n",tid);
    if (idx >= max*max) return;
    // in-place reduction in global memory
    //float *contrast=norm+blockIdx.x*blockDim.x;
    if(idx<size){
        contrast[idx]=norm[idx]*dif[idx];
        //printf("%f %f\n",norm[idx],contrast[idx]);
        __syncthreads();
    }
    for (int stride = 1; stride < max; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            contrast[idx] += contrast[idx+ stride];
            //printf("%d %f\n",idx,contrast[idx]);
        }
        // synchronize within threadblock
        __syncthreads();
    }
    
    if (idx == 0){
        printf("contrast %f\n",contrast[0]);
    }
}

__global__ void calculate_entropy(float *norm,float *entropy,int max,float sum,int size){
    //printf("%d\n",max);
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * max + ix;
    //printf("%d\n",idx);
    int tid=threadIdx.x;
    if(idx<size && norm[idx] !=0){
        entropy[idx]=-(norm[idx]*log10f(norm[idx]));
        //printf("%d f3 %f \n",idx,entropy[idx]);
        __syncthreads();
    }
    for (int stride = 1; stride < size; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            entropy[idx] += entropy[idx+ stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }
    
    if (idx == 0){

        printf("entropy %f\n",entropy[0]);
    }
}

__global__ void calculate_idm(float *norm,float *idm,int*dif,int max,float sum,int size){
    //printf("%d\n",max);
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * max + ix;
    //printf("%d\n",idx);
    int tid=threadIdx.x;
    if(idx<size){
        idm[idx]=((float(1)/(1+dif[idx]))*(norm[idx]));
        //printf("%d  %f %f %f\n",idx,idm[idx],norm[idx],(float(1)/(1+dif[idx])));
        __syncthreads();
    }
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idm[idx] += idm[idx+ stride];
            //printf("%d %f\n",idx,idm[idx]);
        }
        // synchronize within threadblock
        __syncthreads();
    }
    
    if (idx == 0){

        printf("idm %f\n",idm[0]);
    }
}

__global__ void calculate_correlation(float *norm,float *corelation,float *miu_x,float *miu_y,float *stdx,float *stdy,int *ikj,float *dif_variance,int max,float sum,int size){
    //printf("%d\n",max);
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * max + ix;
    int tid=threadIdx.x;
    int i;
    for(i=0;i<max;i++){
        if(idx>=i*max && idx<(i+1)*(max)){
            miu_x[idx]=i*norm[idx];
            //printf("%d,i %d  %f %f \n",idx,i,miu_x[idx],norm[idx]);
        }
        
        //printf("xx %d %f\n",idx*i+idx,miu_x[idx]);
    }
    int blok=0;
    for(i=0;i<max;i++){
        if(blok==i && idx<max){
            miu_y[blok*max+idx]=i*norm[idx*max+i];
            //printf("%d %d,i %d  %f %f %d \n",idx,idx,i,miu_y[idx],norm[idx*max+i],idx*max+i);
            blok++;
        }
        
        //printf("xx %d %f\n",idx*i+idx,miu_x[idx]);
    }
    for(i=0;i<max;i++){
        if(idx>=i*max && idx<(i+1)*(max)){
            stdx[idx]=((i-miu_x[0])*(i-miu_x[0]))*norm[idx];
            //printf("%d,i %d  %f %f \n",idx,i,miu_x[idx],norm[idx]);
    }
        
        //printf("xx %d %f\n",idx*i+idx,miu_x[idx]);
    }
    int batas=0;
    for(i=0;i<max;i++){
       // printf("%d",batas);
        if(batas==i && idx<max){
            stdy[batas*max+idx]=((i-miu_y[0])*(i-miu_y[0]))*norm[idx*max+i];
            //printf("%d %d,i %d  %f %f %d \n",idx,idx,i,stdy[idx],norm[idx*max+i],idx*max+i);
            batas++;
        }   
        
        //printf("xx %d %f\n",idx*i+idx,miu_x[idx]);
    }
    if(idx==0){
        for(i=0;i<max;i++){
            for(int j=0;j<max;j++){
                 ikj[max*i+j]=i*j;
                 //printf("tid %d %d\n",max*i+j,ikj[max*i+j]);
            }
         }
    }
    if(idx<size){
        corelation[idx]=((ikj[idx]*norm[idx]));
        //printf("%d %d,i %d  %f %f \n",idx,idx,i,corelation[idx],norm[idx]);
    }
    for (int stride = 1; stride < size; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            corelation[idx] += corelation[idx+ stride];
            //printf("%d %f\n",idx,corelation[idx]);
        }
        // synchronize within threadblock
        __syncthreads();
    }
    for (int stride = 1; stride < size; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {

            miu_x[idx] += miu_x[idx+ stride];
            stdy[idx] += stdy[idx+ stride];
            miu_y[idx] += miu_y[idx+ stride];
            stdx[idx] += stdx[idx+ stride];
           // corelation[idx] += corelation[idx+ stride];
            //printf("%d %f\n",idx,miu_x[idx]);
        }
        // synchronize within threadblock
        __syncthreads();
    }
    int k=0;
    if(idx==0){
        for(i=0;i<max;i++){
            for(int j=0;j<max;j++){
                k=abs(i-j);
                dif_variance[k]=((k-((miu_x[0]+miu_y[0])/2))*(k-((miu_x[0]+miu_y[0])/2)))*norm[k];

                if(k=i){
                    dif_variance[k]+=dif_variance[i];
                    //printf("%d %f %f %f \n",k,dif_variance[k],(k-((miu_x[0]+miu_y[0])/2)),norm[k]);

            }
            }
         }
         
    }

    for (int stride = 1; stride < size; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            dif_variance[idx] +=dif_variance[idx+stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }
    if (idx == 0){

        printf("correlation %f\n",abs(corelation[0]-miu_x[0]*miu_y[0])/stdx[0]*stdy[0]);
        printf("variance %f\n",stdx[0]);
        printf("difference variance %f\n",dif_variance[0]);
    }
}

__global__ void calculate_sumaverage(float *norm,float *saverage,int max,float sum,int size){
    //printf("%d\n",max);
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * max + ix;
    int tid=threadIdx.x;
    int i;
    for(i=0;i<max;i++){
        if(idx>=i*max && idx<(i+1)*(max)){
            saverage[idx]=i*norm[idx];
            //printf("%d,i %d  %f %f \n",idx,i,saverage[idx],norm[idx]);
        }
        
        //printf("xx %d %f\n",idx*i+idx,miu_x[idx]);
    }
    for (int stride = 1; stride < size; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            saverage[idx] += saverage[idx+ stride];
            //printf("%d %f\n",idx,saverage[idx]);
        }
        // synchronize within threadblock
        __syncthreads();
    }
    
    if (idx == 0){

        printf("sum average %f\n",saverage[0],idx);
    }
}

__global__ void calculate_sumentropy(float *norm,float *sentropy,float *svariance,int max,float sum,int size){
    //printf("%d\n",max);
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * max + ix;
    int tid=threadIdx.x;
    
    if(idx>1 && idx<2*max && norm[idx] !=0){
        sentropy[idx]=-(norm[idx]*log10f(norm[idx]));
        //printf("%d f3 %f \n",idx,entropy[idx]);
        __syncthreads();
    }
    for (int stride = 1; stride < size; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            sentropy[idx] += sentropy[idx+ stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }
    

    if(idx>1 && idx<=2*(max-1)){
        svariance[idx]=((idx-sentropy[0])*(idx-sentropy[0]))*norm[idx];
        //printf("%d f3 %f %f %f\n",idx,svariance[idx],idx-sentropy[0],norm[idx]);
        __syncthreads();
    }
    for (int stride = 1; stride < size; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            svariance[idx] += svariance[idx+ stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }

    if (idx == 0){

        printf("sum entropy %f\n",sentropy[0],idx);
        printf("sum variance %f\n",svariance[0],idx);
    }
}

__global__ void calculate_diffentropy(float *norm,float *dif_entropy,int max,float sum,int size){
    //printf("%d\n",max);
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * max + ix;
    int tid=threadIdx.x;
    if(idx<=(max-1) && norm[idx]>0){
        dif_entropy[idx]=-(norm[idx]*log10f(norm[idx]));
        //printf("%d f3 %f %f %f\n",idx,dif_entropy[idx],norm[idx]);
        __syncthreads();
    }
    for (int stride = 1; stride < size; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            dif_entropy[idx] += dif_entropy[idx+ stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }
    
    if (idx == 0){

        printf("difference entropy%f\n",dif_entropy[0],idx);
    }
}

__global__ void calculate_IMC(float *norm,float *IMC,float *HX,float *HY,float *entropy,float *px,float *py,float *HXY,int max,float sum,int size){
    //printf("%d\n",max);
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * max + ix;
    int tid=threadIdx.x;

   int i;
    for(i=0;i<max;i++){
        if(idx>=i*max && idx<(i+1)*(max) && norm[idx]>0){
            HX[idx]=-(norm[idx]*log10f(norm[idx]));
            //printf("%d,i %d  %f %f \n",idx,i,miu_x[idx],norm[idx]);
    }
    }

    if(idx<size && norm[idx] !=0){
        entropy[idx]=-(norm[idx]*log10f(norm[idx]));
        //printf("%d f3 %f \n",idx,entropy[idx]);
        __syncthreads();
    } 


   
    // for(i=0;i<max;i++){
    //     if(idx>=i*max && idx<(i+1)*(max) && norm[idx]>0){
    //         px[idx]=norm[idx];
    //         //printf("%d,i %d  %f %f \n",idx,i,miu_x[idx],norm[idx]);
    // }
    // }
if(idx<size){
    px[idx]=norm[idx];
}

    int c=0;
    for(i=0;i<max;i++){
        // printf("%d",batas);
         if(c==i && idx<max){
             py[c*max+idx]=norm[idx*max+i];
             //printf("%d %d,i %d  %f %f %d \n",idx,idx,i,stdy[idx],norm[idx*max+i],idx*max+i);
             c++;
         }   
         
         //printf("xx %d %f\n",idx*i+idx,miu_x[idx]);
     }


    int b=0;
    for(i=0;i<max;i++){
        // printf("%d",batas);
         if(b==i && idx<max &&norm[idx*max+i]>0){
             HY[b*max+idx]=-(norm[idx*max+i]*log10f(norm[idx*max+i]));
             //printf("%d %d,i %d  %f %f %d \n",idx,idx,i,HY[b*max+idx],norm[idx*max+i],b*max+i);
             b++;
         }   
         
         //printf("xx %d %f\n",idx*i+idx,miu_x[idx]);
     }
     
     


    for (int stride = 1; stride < size; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            HX[idx] += HX[idx+ stride];
            HY[idx] += HY[idx+ stride];
            px[idx] += px[idx+ stride];
            py[idx] += py[idx+ stride];
            entropy[idx] += entropy[idx+ stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }


    if(idx>9000){
        HXY[idx]=abs(norm[idx]*(log10f((px[0]*py[0]))));
        //printf("tid %d %f %f %f %f \n",idx,HXY[idx],px[0],py[0],norm[idx]);
    }

    for (int stride = 1; stride < size; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            HXY[idx] += HXY[idx+ stride];
            
           
        }
        // synchronize within threadblock
        __syncthreads();
    }
    
    if (idx == 0){
        if(HX[0]>HY[0]){
            IMC[0]=(entropy[0]-HXY[0])/HX[0];
            //printf("x%f %f %f %f px%f %f\n",abs(IMC[0]),entropy[0],HXY[0],HX[0],px[0],py[0]);
        }
        else{
            IMC[0]=entropy[0]-HXY[0]/HY[0];
            //printf("y%f %f %f %f\n",abs(IMC[0]),entropy[0],HXY[0],HY[0]);
        }
        printf("IMC %f\n",abs(IMC[0]));
    }
}

__global__ void calculate_ASM(float *norm,float *ASM,int max,float sum,int size){
    //printf("%d\n",max);
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * max + ix;
    int tid=threadIdx.x;
    if(idx<size){
        ASM[idx]=norm[idx]*norm[idx];
       // printf("%d asm %f\n",idx,norm[idx]);
    }
    //corelation[idx]=(((i*j)*norm[idx]));
            
    for (int stride = 1; stride < size; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            
            ASM[idx] += ASM[idx+stride];
            //printf("%d %f %f\n",idx,corelation[idx],ASM[idx]);
        }
        // synchronize within threadblock
        __syncthreads();
    }
    
    if (idx == 0){

        printf("ASM %f %d\n",ASM[0],idx);
    }
}

void takeimagevalue(const char* filename, rgb_image *img)
{
   
     unsigned error;
     unsigned char* png;
     size_t pngsize;;
   
     lodepng_load_file(&png, &pngsize, filename);
     error = lodepng_decode32(&img->image, &img->width, &img->height, png, pngsize);
    
     if(error) printf("error %u: %s\n", error, lodepng_error_text(error));	
	
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
    char *d;
    long deg =strtol(argv[2],&d,10);
    int degree=deg;
    printf("%s %d degre Starting...\n", argv[0],degree);
    const char* filename = argc > 1 ? argv[1] : "test.png";
    rgb_image img;
    takeimagevalue(filename, &img);
    transformToGrayCuda(&img);
    int nx =img.width;
    int ny =img.height;
    
    printf("image size : %d x %d\n",nx,ny);
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
    printMatrixGambar(h_A,nx,ny);
    printf("gray level :%d \n",max);
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

    //Feature
 
    float *h_contrast;
    h_contrast = (float *)malloc(kBytes);
    float *d_contrast;
    cudaMalloc((void **)&d_contrast, kBytes);
    //for (int i = 0; i < (max+1)*(max+1); i++)
    //{
    //    h_contrast[i] = 0.0;
    //}
    cudaMemcpy(d_contrast, h_contrast, kBytes, cudaMemcpyHostToDevice);
    

    float *h_ASM;
    h_ASM = (float *)malloc(kBytes);
    float *d_ASM;
    cudaMalloc((void **)&d_ASM, kBytes);
    cudaMemcpy(d_ASM, h_ASM, kBytes, cudaMemcpyHostToDevice);

    float *h_entropy,*h_IDM,*h_miux,*h_miuy,*h_stdx,*h_stdy,*h_corelation,*h_variance,*h_saverage,*h_sentropy,*h_svariance,*h_difentropy,*h_HX,*h_HY,*h_IMC,*h_difvariance;
    float *d_entropy,*d_IDM,*d_miux,*d_miuy,*d_stdx,*d_stdy,*d_corelation,*d_variance,*d_saverage,*d_sentropy,*d_svariance,*d_difentropy,*d_HX,*d_HY,*d_IMC,*d_difvariance;
    int*h_ikj,*d_ikj;
    float *h_px,*h_py,*d_px,*d_py,*h_HXY,*d_HXY;
    h_ikj=(int *)malloc(kBytes);
    h_entropy = (float *)malloc(kBytes);
     h_IDM = (float *)malloc(kBytes);
     h_miux = (float *)malloc(kBytes);
     h_miuy = (float *)malloc(kBytes);
     h_stdx = (float *)malloc(kBytes);
     h_stdy = (float *)malloc(kBytes);
     h_corelation = (float *)malloc(kBytes);
     h_variance = (float *)malloc(kBytes);
     h_saverage = (float *)malloc(kBytes);
     h_sentropy = (float *)malloc(kBytes);
     h_svariance = (float *)malloc(kBytes);
     h_difentropy = (float *)malloc(kBytes);
     h_HX = (float *)malloc(kBytes);
     h_HY = (float *)malloc(kBytes);
     h_IMC = (float *)malloc(kBytes);
     h_difvariance = (float *)malloc(kBytes);   
     h_ikj = (int *)malloc(kBytes);   
     h_px = (float *)malloc(kBytes);
     h_py = (float *)malloc(kBytes); 
     h_HXY = (float *)malloc(kBytes); 
    


    cudaMalloc((void **)&d_entropy, kBytes);
    cudaMalloc((void **)&d_IDM, kBytes);
    cudaMalloc((void **)&d_miux, kBytes);
    cudaMalloc((void **)&d_miuy, kBytes);
    cudaMalloc((void **)&d_stdx, kBytes);
    cudaMalloc((void **)&d_stdy, kBytes);
    cudaMalloc((void **)&d_corelation, kBytes);
    cudaMalloc((void **)&d_variance, kBytes);
    cudaMalloc((void **)&d_saverage, kBytes);
    cudaMalloc((void **)&d_sentropy, kBytes);
    cudaMalloc((void **)&d_svariance, kBytes);
    cudaMalloc((void **)&d_difentropy, kBytes);
    cudaMalloc((void **)&d_HX, kBytes);
    cudaMalloc((void **)&d_HY, kBytes);
    cudaMalloc((void **)&d_IMC, kBytes);
    cudaMalloc((void **)&d_difvariance, kBytes);
    cudaMalloc((void **)&d_ikj, kBytes);
    cudaMalloc((void **)&d_px, kBytes);
    cudaMalloc((void **)&d_py, kBytes);
    cudaMalloc((void **)&d_HXY, kBytes);

    cudaMemcpy(d_entropy, h_entropy, kBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_IDM, h_IDM, kBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_miux, h_miux, kBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_miuy, h_miuy, kBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stdx, h_stdx, kBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stdy, h_stdy, kBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_corelation, h_corelation, kBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_variance, h_variance, kBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_saverage, h_saverage, kBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sentropy, h_sentropy, kBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_svariance, h_svariance, kBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_difentropy, h_difentropy, kBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_HX, h_HX, kBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_HY, h_HY, kBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_IMC, h_IMC, kBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_difvariance, h_difvariance, kBytes, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_ikj, h_ikj, kBytes, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_px, h_px, kBytes, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_py, h_py, kBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_HXY, h_HXY, kBytes, cudaMemcpyHostToDevice); 


    // iniitialize host matrix glcm Normalization with integer
    for (int i = 0; i < (max+1)*(max+1); i++)
    {
        h_glcmNorm[i] = 0.0;
    }

    //Penampung nilai glcm di Host
    int *gpuRef;
    gpuRef = (int *)malloc(kBytes);
    memset(gpuRef, 0, kBytes);

    //penampung  nilaiglcm_normalisasi
    int NBytes = (max+1) * (max+1) * sizeof(int);
    float *normhost;
    normhost = (float *)malloc(NBytes);
    memset(normhost, 0, NBytes);
    
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
    // invoke the kernel block size = ny and thread size nx, so its mean there is nx threads/block metode (1xn)
    if(degree==0){
        glcm_calculation_nol<<<ny,nx>>>(d_MatA,d_glcm, nx, ny,max+1);
        cudaMemcpy(gpuRef, d_glcm, kBytes, cudaMemcpyDeviceToHost);
        printMatrixGlcm(gpuRef,max,degree);
    }
    else if(degree ==180){
        glcm_calculation_180<<<ny,nx>>>(d_MatA,d_glcm, nx, ny,max+1);
        cudaMemcpy(gpuRef, d_glcm, kBytes, cudaMemcpyDeviceToHost);
        printMatrixGlcm(gpuRef,max,degree);
    }
    else if(degree==270){
        dim3 block(1, nx);
        dim3 grid((ny + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    
        glcm_calculation_270<<<grid,block>>>(d_MatA,d_glcm, nx, ny,max+1);
        cudaMemcpy(gpuRef, d_glcm, kBytes, cudaMemcpyDeviceToHost);
        printMatrixGlcm(gpuRef,max,degree);
    }
    else if(degree==90){
        dim3 block(1, nx);
        dim3 grid((ny + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    
        glcm_calculation_90<<<grid,block>>>(d_MatA,d_glcm, nx, ny,max+1);
        cudaMemcpy(gpuRef, d_glcm, kBytes, cudaMemcpyDeviceToHost);
        printMatrixGlcm(gpuRef,max,degree);
    }
    else if(degree==45){
        glcm_calculation_45<<<ny,nx>>>(d_MatA,d_glcm, nx, ny,max+1);
        cudaMemcpy(gpuRef, d_glcm, kBytes, cudaMemcpyDeviceToHost);
        printMatrixGlcm(gpuRef,max,degree);
    }
    else if(degree==135){
        glcm_calculation_135<<<ny,nx>>>(d_MatA,d_glcm, nx, ny,max+1);
        cudaMemcpy(gpuRef, d_glcm, kBytes, cudaMemcpyDeviceToHost);
        printMatrixGlcm(gpuRef,max,degree);
    }
    else if(degree==225){
        glcm_calculation_225<<<ny,nx>>>(d_MatA,d_glcm, nx, ny,max+1);
        cudaMemcpy(gpuRef, d_glcm, kBytes, cudaMemcpyDeviceToHost);
        printMatrixGlcm(gpuRef,max,degree);
    }
    else if(degree==315){
        glcm_calculation_315<<<ny,nx>>>(d_MatA,d_glcm, nx, ny,max+1);
        cudaMemcpy(gpuRef, d_glcm, kBytes, cudaMemcpyDeviceToHost);
        printMatrixGlcm(gpuRef,max,degree);
    }


    int sum;
    sum=0;
    for(int i=0;i<((max+1)*(max+1));i++){
        sum +=gpuRef[i];
    }

    normalization<<<max,max>>>(d_glcm,d_glcmNorm,max,sum);
    //copy from device to host
    cudaMemcpy(normhost, d_glcmNorm, kBytes, cudaMemcpyDeviceToHost);
    
    float sums;
    sums=0;
    for(int i=0;i<((max+1)*(max+1));i++){
        sums  +=normhost[i];
    }


    int *dif;
    dif = (int *)malloc(kBytes);
    

    for(int i=0;i<max+1;i++){
        for(int j=0;j<max+1;j++){
            dif[(max+1)*i+j]=(i-j)*(i-j);
            //printf("%d %d\n",(max+1)*i+j,dif[(max+1)*i+j]);
        }
        //printf("\n");
    }
    int *d_dif;
    (cudaMalloc((void **)&d_dif, nBytes));

    // transfer data from host to device
    (cudaMemcpy(d_dif, dif, kBytes, cudaMemcpyHostToDevice));

    int size=(max+1)*(max+1);
    
    
    calculate_contrast<<<max+1,max+1>>>(d_glcmNorm,d_contrast,d_dif,max+1,sums,size);
    calculate_entropy<<<max+1,max+1>>>(d_glcmNorm,d_entropy,max+1,sums,size);
    calculate_idm<<<max+1,max+1>>>(d_glcmNorm,d_IDM,d_dif,max+1,sums,size);
    calculate_correlation<<<max+1,max+1>>>(d_glcmNorm,d_corelation,d_miux,d_miuy,d_stdx,d_stdy,d_ikj,d_difvariance,max+1,sums,size);
   //calculate_difvariance<<<max+1,max+1>>>(d_glcmNorm,d_difvariance,max+1,sums,size);
    calculate_sumaverage<<<max+1,max+1>>>(d_glcmNorm,d_saverage,max+1,sums,size);
    calculate_sumentropy<<<max+1,max+1>>>(d_glcmNorm,d_sentropy,d_svariance,max+1,sums,size);
    //calculate_sumvariance<<<max+1,max+1>>>(d_glcmNorm,d_svavriance,max+1,sums,size);
    calculate_diffentropy<<<max+1,max+1>>>(d_glcmNorm,d_difentropy,max+1,sums,size);
    calculate_ASM<<<max+1,max+1>>>(d_glcmNorm,d_ASM,max+1,sums,size);
    calculate_IMC<<<max+1,max+1>>>(d_glcmNorm,d_IMC,d_HX,d_HY,d_entropy,d_px,d_py,d_HXY,max+1,sums,size);

    end = clock();
    t = ((double) (end - start))/CLOCKS_PER_SEC;

    printMatrixNormalization(normhost,max,degree);

    printf("matrix gambar disimpan di matrixgambar.txt\n");
    printf("matrix glcm disimpan di matrix_glcm_%d.txt\n",degree);
    printf("matrix glcm normalisasi disimpan di matrix_ormalisasi_%d.txt\n",degree);
    
     
    printf("waktu eksekusi: %f\n",t);
    // free host and devide memory
    cudaFree(d_MatA);cudaFree(d_glcm);cudaFree(d_glcmNorm);
    free(h_A);

    // reset device
   cudaDeviceReset();

    return (0);
}