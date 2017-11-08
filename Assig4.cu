#include <stdio.h>
#include <cuda_runtime.h>

#define threadsPerBlock 512

//Device code
 __global__ void calculateCCoeff(const int* AdjMatrix, int numElements, float* globalSum) 
{

    __shared__ float local[threadsPerBlock];
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
	
    if(i < numElements)
    {
		int degree, edgeCount = 0; 
        float coeff = 0;
        
        for(int j=0; j<numElements; j++)
		{		
			//If they are connected
            if(AdjMatrix[i*numElements + j] == 1)
			{			
                degree++;
                for(int k=j+1; k<numElements; k++)
				{
						edgeCount += (AdjMatrix[i*numElements + k] == 1 && AdjMatrix[j*numElements + k] == 1) ? 1 : 0; //If they are neighbors, but not connected
                }
            }
        }
        coeff = degree >=2 ? (2.0f * edgeCount) / (degree * (degree - 1)) : 0; //Calculate the CC
		
        local[threadIdx.x] = coeff;
    }

    __syncthreads();

	//Send sum of threads back to host. Will divide sum by numElements
    if(threadIdx.x == 0 ) {
		//Use thread 0 to calculate total sum
        float sum = 0;
        for( int i = 0; i < threadsPerBlock; i++ )
        {
            int currentIndex = (threadIdx.x + blockIdx.x * blockDim.x) + i;
            sum += currentIndex < numElements ? local[i] : 0;
        }

        atomicAdd(&globalSum[0],sum);
    }
}

 int main(int argc, char* argv[]){
    

    if(argc < 2){
        printf("Argument format must be: ./compiledCode inputFile.txt\n");
        exit(1);
    }

	
    char* inputFile = argv[1];
    FILE* file = fopen(inputFile, "r");
    int u, v;
    int maxNode = 0;
    fscanf(file, "%d %d", &u, &v);
	
	//Find max node
    while(!feof(file)){
        maxNode = u > maxNode ? u : 0;
        maxNode = v > maxNode ? v : 0;

        fscanf(file, "%d %d", &u, &v);
    }

    int n = maxNode + 1;
	
    printf("Total number of elements: %d\n", n);
	
	fclose(file);
	
	//Unsigned value of size of adj matrix
    size_t amSize = n * n * sizeof(int);
	
	//Allocate memory for n * n vector given size
    int* adjVector = (int*) malloc(amSize);    
	
    file = fopen(inputFile, "r");

	int index = 0;
    fscanf(file, "%d %d", &u, &v);
    while(!feof(file)){
        index = u*n + v;
        adjVector[index] = 1;
		
        index = v*n + u;
        adjVector[index] = 1;

        fscanf(file, "%d %d", &u, &v);
    }
    fclose(file);
	
	cudaError_t err = cudaSuccess;

    int* d_adjMatrix = NULL;
    err = cudaMalloc((void **)&d_adjMatrix, amSize);

    if (err != cudaSuccess){
        fprintf(stderr, "Could not allocate memory for Adjacency Matrix (error %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	//Copy host matrix of size 'amSize' to GPU memory
    err = cudaMemcpy(d_adjMatrix, adjVector, amSize, cudaMemcpyHostToDevice);

    if (err != cudaSuccess){
        fprintf(stderr, "Could not copy Adjacency Matrix to device memory (error %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA device with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    float *globalCC = NULL;

    err = cudaMalloc((void **)&globalCC, sizeof(float));
    if (err != cudaSuccess){
        fprintf(stderr, "Could not allocate memory for global Clustering Coefficient variable (error %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaMemset(globalCC, 0, sizeof(float));

	//Calling device method "calculateCCoeff" from host
    calculateCCoeff<<<blocksPerGrid, threadsPerBlock>>>(d_adjMatrix, n, globalCC);
	
    err = cudaGetLastError();

    if (err != cudaSuccess){
        fprintf(stderr, "Could not call device code kernel (error %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
    float hostGlobalCC = 0;
	
    err = cudaMemcpy(&hostGlobalCC, globalCC, sizeof(float), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess){
        fprintf(stderr, "Could not copy global Clustering Coefficient variable back from device (error %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
    hostGlobalCC = hostGlobalCC / n;
	
	printf("\n\nTotal Clustering Coefficient: %f\n\n", hostGlobalCC);
    err = cudaFree(d_adjMatrix);

    if (err != cudaSuccess){
        fprintf(stderr, "Error freeing Adjacency Matrix from device (error %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    free(adjVector);
	
    err = cudaDeviceReset();

    if (err != cudaSuccess){
        fprintf(stderr, "Error freeing local Adjacency Matrix (error %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
    return 0;
}
