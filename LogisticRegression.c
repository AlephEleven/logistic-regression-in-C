#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#define ARRAY_BUFFER 2048
#define MAT_2D 2

#define NORM_MEAN 0
#define NORM_STDEV 1


typedef struct matrix {
    double* data;
    int* shape;
    int shape_len;
    int size;
} matrix;



#define GEN_MATRIX(DATA_NAME, SHAPE_NAME, MAT_NAME, MAT_SHAPEX, MAT_SHAPEY) double DATA_NAME[ARRAY_BUFFER];\
                             int SHAPE_NAME[2] = {MAT_SHAPEX, MAT_SHAPEY}; \
                             matrix MAT_NAME = {.data=DATA_NAME, .shape=SHAPE_NAME}; \
                             update_mat(&MAT_NAME, MAT_2D);

/*

Randomizers + Inits

*/

double random_uniform() { 
    return 2*((double)rand() / (double)RAND_MAX)-1; 
}

double random_gaussian(double mean, double stdev){
    double s = -1;
    double x;
    double y;

    while(s >= 1 || s < 0){
        x = random_uniform();
        y = random_uniform();
        s = x*x + y*y;
    }

    return mean + stdev*(x*sqrt((-2*log(s))/s));
}

void* mat_randn(matrix mat){
    for(int i = 0; i < mat.size; i++){
        mat.data[i] = random_gaussian(NORM_MEAN, NORM_STDEV);
    }
}

void* mat_ones(matrix mat){
    for(int i = 0; i < mat.size; i++){
        mat.data[i] = 1;
    }  
}

void* mat_zeros(matrix mat){
    for(int i = 0; i < mat.size; i++){
        mat.data[i] = 0;
    }  
}

void* update_mat(matrix* mat, int shape_len){
    int mat_size = 1;

    for(int i = 0; i < shape_len; i++){
        mat_size *= mat->shape[i];
    }

    mat->shape_len = shape_len;
    mat->size = mat_size;
}


/*

Index + Printing

*/

int index_2d(int x, int y, int* shape){
    return y + x*shape[1];
}


void* mat_getsubmatrix(matrix matA, matrix matB, int rowStart, int rowEnd, int colStart, int colEnd){
    int indB = 0;
    for(int i = rowStart; i < rowEnd; i++){
        for(int j = colStart; j < colEnd; j++){
            int indA = index_2d(i, j, matA.shape);
            matB.data[indB] = matA.data[indA]; 
            indB++;
        }
    }
}

void print_matrix(matrix mat){
    for(int i = 0; i < mat.shape[0]; i++){
        for(int j = 0; j < mat.shape[1]; j++){
            int ind = index_2d(i, j, mat.shape);
            printf("%f ", mat.data[ind]); 
        }
        printf("\n");
    }
}

void info_matrix(matrix mat){
    printf("shape=(%d, %d), size=%d, dim=%d\n", mat.shape[0], mat.shape[1], mat.size, mat.shape_len);
}

/*

Math Operations

*/

double mat_sum(matrix mat){
    double sum = 0;
    for(int i = 0; i < mat.shape[0]*mat.shape[1]; i++){
        sum += mat.data[i];
    }

    return sum;
}

double mat_prod(matrix mat){
    double sum = 1;
    for(int i = 0; i < mat.shape[0]*mat.shape[1]; i++){
        sum *= mat.data[i];
    }

    return sum;
}

void* mat_transpose(matrix matA, matrix matB){
    for(int i = 0; i < matA.shape[1]; i++){
        for(int j = 0; j < matA.shape[0]; j++){
            int indA = index_2d(j, i, matA.shape);
            int indB = index_2d(i, j, matB.shape);
            matB.data[indB] = matA.data[indA];
        }
    }  
}

double mat_sum_axis(matrix mat, int index, int axis){
    double sum = 0;
    if(axis == 0){
        for(int i = 0; i < mat.shape[1]; i++){
           int ind = index_2d(index, i, mat.shape);
           sum += mat.data[ind];
        }
        return sum;
    }
    if(axis == 1){
        for(int j = 0; j < mat.shape[0]; j++){
           int ind = index_2d(j, index, mat.shape);
           sum += mat.data[ind];
        }
        return sum;
    }
    return -1;
}

void* mat_mul(matrix matA, matrix matB, matrix matC){
    int row = matA.shape[0];
    int col = matB.shape[1];
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            
            int indC = index_2d(i, j, matC.shape);
            
            for(int c = 0; c < matA.shape[1]; c++){
                int indA = index_2d(i, c, matA.shape);
                int indB = index_2d(c, j, matB.shape);
                matC.data[indC] += matA.data[indA]*matB.data[indB];
            }

        }
    }
}

void* mat_add(matrix matA, matrix matB, matrix matC){
    for(int i = 0; i < matA.size; i++){
        matC.data[i] = matA.data[i]+matB.data[i];
    }
}

void* mat_sub(matrix matA, matrix matB, matrix matC){
    for(int i = 0; i < matA.size; i++){
        matC.data[i] = matA.data[i]-matB.data[i];
    }
}

void* mat_scale(matrix mat, double scale){
    for(int i = 0; i < mat.size; i++){
        mat.data[i] = mat.data[i]*scale;
    }
}

void* mat_apply(matrix mat, double (*f)(double)){
    for(int i = 0; i < mat.size; i++){
        mat.data[i] = f(mat.data[i]);
    }
}

double sigmoid(double z){
    return 1/(1 + exp(-z));
}

double MSL(matrix matA, matrix matB){
    double msl = 0;
    for(int i = 0; i < matA.size; i++){
        msl += pow((matA.data[i]-matB.data[i]), 2);
    }

    return msl/(double)matA.size;
}

void* MSL_dx(matrix matA, matrix matB, matrix matC){
    for(int i = 0; i < matA.size; i++){
        matC.data[i] = (-2/(float)matA.size) * (matA.data[i]-matB.data[i]);
    }


}


void* mat_linear(matrix matW, matrix matx, matrix matb, matrix matz){
    /*
    z = Wx + b
    */
    mat_mul(matW, matx, matz);
    mat_add(matz, matb, matz);
}

void* mat_lineartransform(matrix matW, matrix matx, matrix matb, matrix matz, double (*f)(double)){
    mat_linear(matW, matx, matb, matz);
    mat_apply(matz, f);
}

double accuracy(matrix pred, matrix true){
    double acc = 0;
    for(int i = 0; i < pred.size; i++){
        if (true.data[i] == 1){
            acc += pred.data[i] >= 0.5;
        }
        else{
            acc += pred.data[i] < 0.5;
        }
    }
    return acc/pred.size;
}


/*

Models

*/


/*
Trains weight+bias pair on data using logistic regression.

Inputs
- X,y: Training data
- W,b: weight+bias to be learnt
- activation: function used for activation
- lr: learning rate
- epochs: total runs through data
*/
void* LogisticRegression(matrix X, matrix y, 
                         matrix W, matrix b,
                         double (*activation)(double),
                         double lr,
                         int epochs){
    
    int N = X.shape[0];
    int in_dim = X.shape[1];


    GEN_MATRIX(current_data, current_data_shape, mat_current_data, 1, in_dim);

    GEN_MATRIX(current_data_T, current_data_T_shape, mat_current_data_T, in_dim, 1);



    int out_dim = y.shape[1];

    GEN_MATRIX(current_label, current_label_shape, mat_current_label, 1, out_dim);

    GEN_MATRIX(z, z_shape, mat_z, b.shape[0], b.shape[1]);


    GEN_MATRIX(loss_dx, loss_dx_shape, mat_loss_dx, 1, out_dim);

    GEN_MATRIX(W_dx, W_dx_shape, mat_W_dx, W.shape[1], W.shape[0]);

    GEN_MATRIX(dw, dw_shape, mat_dw, W.shape[0], W.shape[1]);




    double loss = 0;
    int acc = 0;


    for(int epoch = 1; epoch < epochs+1; epoch++){
        for(int ind = 0; ind < N; ind++){
            mat_getsubmatrix(X, mat_current_data, ind, ind+1, 0, in_dim);
            mat_getsubmatrix(y, mat_current_label, ind, ind+1, 0, out_dim);

            mat_transpose(mat_current_data, mat_current_data_T);

            mat_lineartransform(W, mat_current_data_T, b, mat_z, activation);

            loss += MSL(mat_current_label, mat_z);

            MSL_dx(mat_current_label, mat_z, mat_loss_dx);

            mat_mul(mat_loss_dx, mat_current_data, mat_dw);
            mat_scale(mat_dw, lr);
            mat_sub(W, mat_dw, W);

            mat_scale(mat_loss_dx, lr);
            mat_sub(b, mat_loss_dx, b);

            acc += (int)accuracy(mat_z, mat_current_label);


        }
        printf("Epoch (%d/%d) | Loss = %f, Acc=(%d/%d)\n", epoch, epochs, loss/N, acc, N);
        loss = 0;
        acc = 0;



    }


}



int main(int argc, char** argv){
    srand(time(NULL));


    //Logistic Regression

    GEN_MATRIX(W, W_shape, matW, 1, 4);
    mat_randn(matW);


    GEN_MATRIX(b, b_shape, matb, 1, 1);
    mat_zeros(matb);

    


    double X[ARRAY_BUFFER] = {.1, 3.5, 1.4, 0.2, 4.9, 3. , 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6,
       3.1, 1.5, 0.2, 5. , 3.6, 1.4, 0.2, 5.4, 3.9, 1.7, 0.4, 4.6, 3.4,
       1.4, 0.3, 5. , 3.4, 1.5, 0.2, 4.4, 2.9, 1.4, 0.2, 4.9, 3.1, 1.5,
       0.1, 5.4, 3.7, 1.5, 0.2, 4.8, 3.4, 1.6, 0.2, 4.8, 3. , 1.4, 0.1,
       4.3, 3. , 1.1, 0.1, 5.8, 4. , 1.2, 0.2, 5.7, 4.4, 1.5, 0.4, 5.4,
       3.9, 1.3, 0.4, 5.1, 3.5, 1.4, 0.3, 5.7, 3.8, 1.7, 0.3, 5.1, 3.8,
       1.5, 0.3, 5.4, 3.4, 1.7, 0.2, 5.1, 3.7, 1.5, 0.4, 4.6, 3.6, 1. ,
       0.2, 5.1, 3.3, 1.7, 0.5, 4.8, 3.4, 1.9, 0.2, 5. , 3. , 1.6, 0.2,
       5. , 3.4, 1.6, 0.4, 5.2, 3.5, 1.5, 0.2, 5.2, 3.4, 1.4, 0.2, 4.7,
       3.2, 1.6, 0.2, 4.8, 3.1, 1.6, 0.2, 5.4, 3.4, 1.5, 0.4, 5.2, 4.1,
       1.5, 0.1, 5.5, 4.2, 1.4, 0.2, 4.9, 3.1, 1.5, 0.1, 5. , 3.2, 1.2,
       0.2, 5.5, 3.5, 1.3, 0.2, 4.9, 3.1, 1.5, 0.1, 4.4, 3. , 1.3, 0.2,
       5.1, 3.4, 1.5, 0.2, 5. , 3.5, 1.3, 0.3, 4.5, 2.3, 1.3, 0.3, 4.4,
       3.2, 1.3, 0.2, 5. , 3.5, 1.6, 0.6, 5.1, 3.8, 1.9, 0.4, 4.8, 3. ,
       1.4, 0.3, 5.1, 3.8, 1.6, 0.2, 4.6, 3.2, 1.4, 0.2, 5.3, 3.7, 1.5,
       0.2, 5. , 3.3, 1.4, 0.2, 7. , 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5,
       6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4. , 1.3, 6.5, 2.8, 4.6, 1.5, 5.7,
       2.8, 4.5, 1.3, 6.3, 3.3, 4.7, 1.6, 4.9, 2.4, 3.3, 1. , 6.6, 2.9,
       4.6, 1.3, 5.2, 2.7, 3.9, 1.4, 5. , 2. , 3.5, 1. , 5.9, 3. , 4.2,
       1.5, 6. , 2.2, 4. , 1. , 6.1, 2.9, 4.7, 1.4, 5.6, 2.9, 3.6, 1.3,
       6.7, 3.1, 4.4, 1.4, 5.6, 3. , 4.5, 1.5, 5.8, 2.7, 4.1, 1. , 6.2,
       2.2, 4.5, 1.5, 5.6, 2.5, 3.9, 1.1, 5.9, 3.2, 4.8, 1.8, 6.1, 2.8,
       4. , 1.3, 6.3, 2.5, 4.9, 1.5, 6.1, 2.8, 4.7, 1.2, 6.4, 2.9, 4.3,
       1.3, 6.6, 3. , 4.4, 1.4, 6.8, 2.8, 4.8, 1.4, 6.7, 3. , 5. , 1.7,
       6. , 2.9, 4.5, 1.5, 5.7, 2.6, 3.5, 1. , 5.5, 2.4, 3.8, 1.1, 5.5,
       2.4, 3.7, 1. , 5.8, 2.7, 3.9, 1.2, 6. , 2.7, 5.1, 1.6, 5.4, 3. ,
       4.5, 1.5, 6. , 3.4, 4.5, 1.6, 6.7, 3.1, 4.7, 1.5, 6.3, 2.3, 4.4,
       1.3, 5.6, 3. , 4.1, 1.3, 5.5, 2.5, 4. , 1.3, 5.5, 2.6, 4.4, 1.2,
       6.1, 3. , 4.6, 1.4, 5.8, 2.6, 4. , 1.2, 5. , 2.3, 3.3, 1. , 5.6,
       2.7, 4.2, 1.3, 5.7, 3. , 4.2, 1.2, 5.7, 2.9, 4.2, 1.3, 6.2, 2.9,
       4.3, 1.3, 5.1, 2.5, 3. , 1.1, 5.7, 2.8, 4.1, 1.3};

    int X_shape[2] = {100, 4};
    matrix matX = {.data=X, .shape=X_shape};
    update_mat(&matX, MAT_2D);

    double y[ARRAY_BUFFER] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
    int y_shape[2] = {100, 1};
    matrix maty = {.data=y, .shape=y_shape};
    update_mat(&maty, MAT_2D);


    LogisticRegression(matX, maty, matW, matb, sigmoid, 0.01, 10);

    //Sample Prediction

    double X_sample[ARRAY_BUFFER] = {5.7, 2.8, 4.1, 1.3};
    int X_sample_shape[2] = {4, 1};
    matrix matX_sample = {.data=X_sample, .shape=X_sample_shape};
    update_mat(&matX_sample, MAT_2D);

    double z[ARRAY_BUFFER];
    int z_shape[2] = {1, 1};
    matrix matz = {.data=z, .shape=z_shape};
    update_mat(&matz, MAT_2D);  

    printf("\nInput:\n");
    print_matrix(matX_sample);

    mat_lineartransform(matW, matX_sample, matb, matz, sigmoid);

    printf("\nPredicted: %f | Truth %f\n", matz.data[0], y[99]);

    return 0;
}
