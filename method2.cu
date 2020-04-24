
   #include <cuda_runtime.h>
   #include <cuda.h>
   #include <stdio.h>
   #include <stdlib.h>
   #include <time.h>
   #include "kuda.h"
   #include "lodepng.h"
// Banyak nx * nx Matrix
// Banyak Max * Max Matrix
int Max;

void printMatrixGlcm(int *C, const int Max,int degree)
{
    int *ic = C;
    FILE * fp = NULL;
    if(degree==0){
        fp = fopen("matrix_glcm_0_method_2.txt", "w+");
    }
    else if(degree==90){
        fp = fopen("matrix_glcm_90_method_2.txt", "w+");
    }
    else if(degree==180){
        fp = fopen("matrix_glcm_180_method_2.txt", "w+");
    }
    else if(degree==270){
        fp = fopen("matrix_glcm_270_method_2.txt", "w+");
    }
    else if(degree==45){
        fp = fopen("matrix_glcm_45_method_2.txt", "w+");
    }
    else if(degree==135){
        fp = fopen("matrix_glcm_135_method_2.txt", "w+");
    }
    else if(degree==225){
        fp = fopen("matrix_glcm_225_method_2.txt", "w+");
    }
    else if(degree==315){
        fp = fopen("matrix_glcm_315_method_2.txt", "w+");
    }

    if(fp == NULL){
        printf("Error creating results file\n");
        exit(1);
    }
    for (int iy = 0; iy <Max; iy++)
    {
        for (int ix = 0; ix <Max; ix++)
        {
            fprintf(fp, "%d  ", ic[ix]);
   
        }
        fprintf(fp, "\n\n");
        ic += (Max);

    }
   
    printf("\n");
    fclose(fp);
    return;
}

void printMatrixnxormalization(float *C, const int Max,int degree)
{
    float *ic = C;
    FILE * fp = NULL;
    if(degree==0){
        fp = fopen("matrix_normalisasi_0_method2.txt", "w+");
    }
    else if(degree==90){
        fp = fopen("matrix_normalisasi_90_method_2.txt", "w+");
    }
    else if(degree==180){   
        fp = fopen("matrix_normalisasi_180_method_2.txt", "w+");
    }
    else if(degree==270){
        fp = fopen("matrix_normalisasi_270_method_2.txt", "w+");
    }
    else if(degree==45){
        fp = fopen("matrix_normalisasi_45_method_2.txt", "w+");
    }
    else if(degree==135){
        fp = fopen("matrix_normalisasi_135_method_2.txt", "w+");
    }
    else if(degree==225){
        fp = fopen("matrix_normalisasi_225_method_2.txt", "w+");
    }
    else if(degree==315){
        fp = fopen("matrix_normalisasi_315_method_2.txt", "w+");
    }

    if(fp == NULL){
        printf("Error creating results file\n");
        exit(1);
    }
    for (int iy = 0; iy < Max; iy++)
    {
        for (int ix = 0; ix <Max; ix++)
        {
            
            fprintf(fp, "%.7f  ", ic[ix]);

        }
        fprintf(fp, "\n\n");
        ic +=Max;
        
    }

    printf("\n");
    fclose(fp);
    return;
}

// void calculate_glcm_host(int *matrix,int *glcm,int nx,int ny,int Max){
//     int i,j;
//     for(i=0;i<nx;i++){
//         for(j=0;j<ny;j++){
//             glcm[matrix[i]][matrix[j]] +=1;
//         }
//     }
// }

//calculate glcm
__global__ void Div0(int *matrix , int *newMatrix,int nx,int ny,int Max){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int Index = iy * nx + ix;
    int posisi = 0;

    for(int i = 0 ; i < nx ; i += 2){
        if(Index >= i * nx && Index < ((i + 1) * nx) - 1){

            posisi = matrix[Index] * Max + matrix[Index + 1];
            atomicAdd(&newMatrix[posisi],1);

            posisi = matrix[Index + Max] * Max + matrix[Index + (Max + 1)];
            atomicAdd(&newMatrix[posisi],1);
        }
    }
}

__global__ void Div45(int *matrix , int *newMatrix,int nx,int ny,int Max){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int Index = iy * nx + ix;
    int posisi = 0;

    for(int i = 0 ; i < nx - 1 ; i++){
        if(Index >= i * nx && Index < ((i + 1) * nx) - 1){
        posisi = matrix[Index + nx] * nx + matrix[Index + 1];
        atomicAdd(&newMatrix[posisi],1);
        printf("Index : %d %d\n",Index + nx , Index + 1);
        }
    }
}

__global__ void Div90(int *matrix , int *newMatrix,int nx,int ny,int Max){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int Index = iy * nx + ix;
    int posisi = 0;

    for(int i = 0 ; i < nx - 1 ; ++i){
        if(Index >= i * nx && Index < ((i + 1) * nx) - 1){
            if(Index == 0 || Index % 2 == 0){
                posisi = matrix[Index + nx] * nx + matrix[Index];
                atomicAdd(&newMatrix[posisi],1);

                posisi = matrix[Index + (nx + 1)] * nx + matrix[Index + 1];
                atomicAdd(&newMatrix[posisi],1);
                printf("Index : %d %d dan %d %d\n",Index + nx , Index, Index + (nx + 1),Index + 1);
            }
        }
    }
}

__global__ void Div135(int *matrix , int *newMatrix,int nx,int ny,int Max){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int Index = iy * nx + ix;
    int posisi = 0;

    for(int i = 0 ; i < nx - 1 ; ++i){
        if(Index >= i * nx && Index < ((i + 1) * nx) - 1){

            posisi = matrix[Index + (nx + 1)] * nx + matrix[Index];
            atomicAdd(&newMatrix[posisi],1);
            printf("Index : %d %d\n",Index + (nx + 1), Index);
        }
    }
}

__global__ void Div180(int *matrix , int *newMatrix,int nx,int ny,int Max){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int Index = iy * nx + ix;
    int posisi = 0;

    for(int i = 0 ; i < nx ; i += 2){
        if(Index >= i * nx && Index < ((i + 1) * nx) - 1){
                
                posisi = matrix[Index + 1] * nx + matrix[Index];
                atomicAdd(&newMatrix[posisi],1);

                posisi = matrix[Index + (nx + 1)] * nx + matrix[Index + nx];
                atomicAdd(&newMatrix[posisi],1);
        }
    }
}

__global__ void Div225(int *matrix , int *newMatrix,int nx,int ny,int Max){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int Index = iy * nx + ix;
    int posisi = 0;

    for(int i = 0 ; i < nx - 1 ; ++i){
        if(Index >= i * nx && Index < ((i + 1) * nx) - 1){
            posisi = matrix[Index + 1] * nx + matrix[Index + nx];
            atomicAdd(&newMatrix[posisi],1);
            printf("Index : %d %d\n",Index + 1, Index + nx);
        }
    }
}

__global__ void Div270(int *matrix , int *newMatrix,int nx,int ny,int Max){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int Index = iy * nx + ix;
    int posisi = 0;

    for(int i = 0 ; i < nx - 1 ; ++i){
        if(Index >= i * nx && Index < ((i + 1) * nx) - 1){
            if(Index == 0 || Index % 2 == 0){
                posisi = matrix[Index] * nx + matrix[Index + nx];
                atomicAdd(&newMatrix[posisi],1);

                posisi = matrix[Index + 1] * nx + matrix[Index + (nx + 1)];
                atomicAdd(&newMatrix[posisi],1);
                printf("Index : %d %d dan %d %d\n",Index,Index + nx , Index + 1, Index + (nx + 1));
            }
        }
    }
}

__global__ void Div315(int *matrix , int *newMatrix,int nx,int ny,int Max){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int Index = iy * nx + ix;
    int posisi = 0;

    for(int i = 0 ; i < nx - 1 ;  ++i ){
        if(Index >= i * nx && Index < ((i + 1) * nx) - 1){
            posisi = matrix[Index] * nx + matrix[Index + (nx + 1)];
            atomicAdd(&newMatrix[posisi],1);
            printf("Index : %d %d\n",Index,Index + (nx + 1));
        }
    }
}

__global__ void Mul(float *newMatrix,float *mulMatrix,int Max,float *sumMatrix){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // int Index = iy * nx + ix;

    for (int k = 0; k < Max; k++) {
        // Accumulate results for a single element
        // c[row * nx + col] += a[row * nx + k] * b[k * nx + col];
        // printf("C[%d] = a[%d] * b[%d]\n",row * nx + col,row * nx + k, k * nx + col);
        atomicAdd(&mulMatrix[row * Max + col],newMatrix[row * Max + k] * newMatrix[k * Max + col]);
        // atomicAdd(&sumMatrix[0],mulMatrix[row * Max + col]);
    }
}


__global__ void Jumlah(float *sumMatrix,float *mulMatrix){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index<1) printf("%f",mulMatrix[0]);
    atomicAdd(&sumMatrix[0],mulMatrix[Index]);
    
}

__global__ void normalization(int *glcm,float *norm,int Max,int sum){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * Max + ix;
    __syncthreads();
    if(idx<(Max+1)*(Max+1)){
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

__global__ void calculate_ASM(float *norm,float *ASM,float *mulMatrix,int Max){
    //printf("%d\n",max);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // int Index = iy * N + ix;

    for (int k = 0; k < Max; k++) {
        // Accumulate results for a single element
        // c[row * N + col] += a[row * N + k] * b[k * N + col];
        // printf("C[%d] = a[%d] * b[%d]\n",row * N + col,row * N + k, k * N + col);
        atomicAdd(&mulMatrix[row * Max + col],norm[row * Max + k] * norm[k * Max + col]);
    }
    int Index = blockIdx.x * blockDim.x + threadIdx.x;

    atomicAdd(&ASM[0],mulMatrix[Index]);

    if (Index == 0){

        printf("ASM %f\n",ASM[0]);
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
    unsigned int  width = img->width;
    unsigned int height = img->height;
    int n =width*height; 
    size_t size = n * 4 * sizeof(unsigned char);
	

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

int main(int argc, char *argv[]){


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
     
    int *matrix,*glcm;
    float *norm,*mulMatrix,*sumMatrix;

    cudaMallocManaged(&matrix, (nx * ny) * sizeof(int));

    for(int i = 0 ; i < (nx * nx) ; ++i){
        matrix[i] = img.image[i];
        if(matrix[i] > Max){
            Max = matrix[i];
        }
    }
    
    for(int i = 0 ; i < nx ; ++i){
        for(int j = 0 ; j < nx ; ++j){
           // printf("%4d",matrix[i * nx + j]);
        }
        //printf("\n");
    }
    //printf("\n\n");
    Max = Max + 1; // karena index dimulai dari 0 dan Maximum 3 ( 0 - 3 = 4 ) jadi Max ditambah 1;
    int kBytes = Max * Max * sizeof(float);
    int nBytes =  img.width * img.height * sizeof(float);
    cudaMallocManaged(&glcm, (Max * Max) * sizeof(int));
    cudaMallocManaged(&mulMatrix, (Max * Max) * sizeof(float));
    cudaMallocManaged(&sumMatrix, (Max * Max) * sizeof(float));
    cudaMallocManaged(&norm, (Max * Max) * sizeof(float));
    for(int i = 0 ; i < (Max * Max) ; ++i){
        glcm[i] = 0;
        mulMatrix[i] = 0;
    }

    float *h_contrast;
    h_contrast = (float *)malloc(kBytes);
    float *d_contrast;
    cudaMalloc((void **)&d_contrast, kBytes);
    //for (int i = 0; i < (Max+1)*(Max+1); i++)
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


    dim3 block(2 ,2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    clock_t start, end;
    double t = 0;
    start = clock();
    // invoke kernel for calculation
    if(degree==0){
        Div0<<<grid,block>>>(matrix,glcm, nx, ny,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(glcm,Max,degree);
    }
    else if(degree ==180){
        Div180<<<grid,block>>>(matrix,glcm, nx, ny,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(glcm,Max,degree);
    }
    else if(degree==270){
        Div270<<<grid,block>>>(matrix,glcm, nx, ny,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(glcm,Max,degree);
    }
    else if(degree==90){
        Div90<<<grid,block>>>(matrix,glcm, nx, ny,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(glcm,Max,degree);
    }
    else if(degree==45){
        Div45<<<grid,block>>>(matrix,glcm, nx, ny,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(glcm,Max,degree);
    }
    else if(degree==135){
        Div135<<<grid,block>>>(matrix,glcm, nx, ny,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(glcm,Max,degree);
    }
    else if(degree==225){
        Div225<<<grid,block>>>(matrix,glcm, nx, ny,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(glcm,Max,degree);
    }
    else if(degree==315){
        Div315<<<grid,block>>>(matrix,glcm, nx, ny,Max);
        cudaDeviceSynchronize();
        printMatrixGlcm(glcm,Max,degree);
    }
    cudaDeviceSynchronize();
    int sum;
    sum=0;
    for(int i=0;i<Max*Max;i++){
        sum +=glcm[i];
    }
    printf("sum %d",sum);
    normalization<<<Max,Max>>>(glcm,norm,Max,sum);

    
    cudaDeviceSynchronize();
    printMatrixnxormalization(norm,Max,degree);
    float sums;
    sums=0;
    for(int i=0;i<Max*Max;i++){
        sums  +=norm[i];
    }
    //Jumlah <<< Max,Max >>>(sumMatrix,norm);
    printf("jumlah %f\n",sums);
    int *dif;
    dif = (int *)malloc(kBytes);
    int *d_dif;
    (cudaMalloc((void **)&d_dif, nBytes));

    // transfer data from host to device
    (cudaMemcpy(d_dif, dif, kBytes, cudaMemcpyHostToDevice));

    int size=(Max)*(Max);
    
    
    calculate_contrast<<<Max+1,Max+1>>>(norm,d_contrast,d_dif,Max,sums,size);
    calculate_entropy<<<Max+1,Max+1>>>(norm,d_entropy,Max,sums,size);
    calculate_idm<<<Max+1,Max+1>>>(norm,d_IDM,d_dif,Max,sums,size);
    calculate_correlation<<<Max+1,Max+1>>>(norm,d_corelation,d_miux,d_miuy,d_stdx,d_stdy,d_ikj,d_difvariance,Max,sums,size);
   //calculate_difvariance<<<Max+1,Max+1>>>(d_glcmnxorm,d_difvariance,Max+1,sums,size);
    calculate_sumaverage<<<Max+1,Max+1>>>(norm,d_saverage,Max,sums,size);
    calculate_sumentropy<<<Max+1,Max+1>>>(norm,d_sentropy,d_svariance,Max,sums,size);
    //calculate_sumvariance<<<Max+1,Max+1>>>(norm,d_svavriance,Max+1,sums,size);
    calculate_diffentropy<<<Max+1,Max+1>>>(norm,d_difentropy,Max,sums,size);
    calculate_ASM<<<Max+1,Max+1>>>(norm,d_ASM,mulMatrix,Max);
    calculate_IMC<<<Max+1,Max+1>>>(norm,d_IMC,d_HX,d_HY,d_entropy,d_px,d_py,d_HXY,Max,sums,size);

    end = clock();
    t = ((double) (end - start))/CLOCKS_PER_SEC;

    

    printf("matrix gambar disimpan di matrixgambar.txt\n");
    printf("matrix glcm disimpan di matrix_glcm_%d.txt\n",degree);
    printf("matrix glcm normalisasi disimpan di matrix_ormalisasi_%d.txt\n",degree);
    
     
    printf("waktu eksekusi: %f\n",t);
    // free host and devide memory
    cudaFree(matrix);cudaFree(glcm);cudaFree(norm);
    cudaFree(d_ASM);cudaFree(d_contrast);cudaFree(d_dif);cudaFree(d_difentropy);cudaFree(d_difvariance);
    cudaFree(d_entropy);cudaFree(d_HX);cudaFree(d_HXY);cudaFree(d_HY);cudaFree(d_IDM);
    cudaFree(d_ikj);cudaFree(d_IMC);cudaFree(d_saverage);cudaFree(mulMatrix);
}