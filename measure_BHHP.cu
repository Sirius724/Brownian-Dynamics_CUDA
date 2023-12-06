#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include "../timer.cuh"
#include <math.h>
#include <iostream>
#include <fstream>
#include <curand.h>
#include <curand_kernel.h>
#include <sys/stat.h>
#include "../MT.h"
using namespace std;

//Using "const", the variable is shared into both gpu and cpu. 
const int  NT = 1024; //Num of the cuda threads.
const int  NP = 4000; //Particle number.
const int  NB = (NP+NT-1)/NT; //Num of the cuda blocks.
const int  NN = 500; //nearest neighbor maximum numbers
const double dt = 0.01;
const int timemax = 2e+5;
const int timeeq = 1000;
//Langevin parameters
const double zeta = 1.0;
const double temp = 0.6;
const double rho = 0.8;
const double RCHK= 5.0;
const double rcut= 1.0;

//Initiallization of "curandState"
__global__ void setCurand(unsigned long long seed, curandState *state){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  curand_init(seed, i_global, 0, &state[i_global]);
}

//Gaussian random number's generation
__global__ void genrand_kernel(float *result, curandState *state){  
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  result[i_global] = curand_normal(&state[i_global]);
}

//Gaussian random number's generation
__global__ void langevin_kernel(double*x_dev,double*y_dev,double *vx_dev,double *vy_dev,double *fx_dev,double *fy_dev,curandState *state, double noise_intensity,double LB){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
    //  printf("%d,%f\n",i_global,v_dev[i_global]);
    vx_dev[i_global] += -zeta*vx_dev[i_global]*dt+ fx_dev[i_global]*dt + noise_intensity*curand_normal(&state[i_global]);
    vy_dev[i_global] += -zeta*vy_dev[i_global]*dt+ fy_dev[i_global]*dt + noise_intensity*curand_normal(&state[i_global]);
    x_dev[i_global] += vx_dev[i_global]*dt;
    y_dev[i_global] += vy_dev[i_global]*dt;

    x_dev[i_global]  -= LB*floor(x_dev[i_global]/LB);
    y_dev[i_global]  -= LB*floor(y_dev[i_global]/LB);
    //printf("%f\n",x_dev[i_global]);
  }
}

__global__ void langevin_kernel_nobound(double*x_dev,double*y_dev,double *vx_dev,double *vy_dev,double *fx_dev,double *fy_dev,curandState *state, double noise_intensity,double LB){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
    //  printf("%d,%f\n",i_global,v_dev[i_global]);
    vx_dev[i_global] += -zeta*vx_dev[i_global]*dt+ fx_dev[i_global]*dt + noise_intensity*curand_normal(&state[i_global]);
    vy_dev[i_global] += -zeta*vy_dev[i_global]*dt+ fy_dev[i_global]*dt + noise_intensity*curand_normal(&state[i_global]);
    x_dev[i_global] += vx_dev[i_global]*dt;
    y_dev[i_global] += vy_dev[i_global]*dt;

    //x_dev[i_global]  -= LB*floor(x_dev[i_global]/LB);
    //y_dev[i_global]  -= LB*floor(y_dev[i_global]/LB);
    //printf("%f\n",x_dev[i_global]);
  }
}

__global__ void p_bound(double *x_dev, double *y_dev, double LB){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
    x_dev[i_global]  -= LB*floor(x_dev[i_global]/LB);
    y_dev[i_global]  -= LB*floor(y_dev[i_global]/LB);
    //printf("%f\n",x_dev[i_global]);
  }
}

int calc_com(double *x_corr, double *y_corr, double *corr_x, double *corr_y){
  *corr_x = 0.;
  *corr_y = 0.; 
  for (int i=0; i<NP; i++){
    *corr_x += x_corr[i];
    *corr_y += y_corr[i];
  }
  //printf("%f  %f\n",*corr_x, *corr_y);

  return 0;
}

double calc_MSD(double *MSD_host){
  double msd = 0.;

  for (int i=0; i<NP; i++){
    msd += MSD_host[i];
  }
  return msd;
}

double calc_ISF(double *ISF_host){
  double isf = 0.;

  for (int i=0; i<NP; i++){
    isf += ISF_host[i];
  }

  return isf;
}

double calc_K(double *vx, double*vy){
  double K = 0.;
  for (int i=0; i<NP; i++){
    K += (vx[i]*vx[i]+vy[i]*vy[i])*0.5/(double) NP;
  }
  return K;
}

void save_position(double *x, double*y, double *xi, double *yi, double corr_x, double corr_y){
  for (int i=0; i<NP; i++){
    xi[i] = x[i] - corr_x;
    //if(i%1000==0){printf("%.4f %.4f %.4f\n",*corr_x, x0[i], x[i]);}
    yi[i] = y[i] - corr_y;
  }
  //return 0;
}

__global__ void com_correction(double *x_dev, double *y_dev, double *x_corr_dev, double *y_corr_dev, double LB){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  static double x0[NP], y0[NP];
  static bool IsFirst=true;

  if(i_global<NP){
    if(IsFirst){
      x0[i_global] = x_dev[i_global];
      y0[i_global] = y_dev[i_global];
      IsFirst = false;
    }

    double dx, dy;
    dx = x_dev[i_global] - x0[i_global];
    dy = y_dev[i_global] - y0[i_global];

    dx -= LB*floor(dx/LB+0.5);
    dy -= LB*floor(dy/LB+0.5);

    x_corr_dev[i_global] += dx/NP;
    y_corr_dev[i_global] += dy/NP;
    //if(i_global%1000 == 0){printf("%d %.5f	%.5f\n", i_global, x0[i_global], x_dev[i_global]);}
    x0[i_global] = x_dev[i_global];
    y0[i_global] = y_dev[i_global];
  }
}

/*int com_correction(double *x,double *y,double *corr_x,double *corr_y, double LB){
  int i;
  double dx,dy;
  static double x0[NP],y0[NP];
  static bool IsFirst = true;
  if(IsFirst){
    for(i=0;i<NP;i++){
      x0[i]=x[i];
     y0[i]=y[i];
    }
    IsFirst = false;
  }

  for(i=0;i<NP;i++){
    dx=x[i]-x0[i];
    if(dx>0.5*LB) dx-=LB;
    else if(dx<-0.5*LB)dx+=LB;

    dy=y[i]-y0[i];
    if(dy>0.5*LB) dy-=LB;
    else if(dy<-0.5*LB)dy+=LB;

   *corr_x+=dx/NP; //center of mass displacement.x
   *corr_y+=dy/NP;

   x0[i]=x[i];
   y0[i]=y[i];
  }
  return 0;
}
*/

__global__ void ini_gate_kernel(int *gate_dev,int c)
{
  gate_dev[0]=c;
}

__global__ void disp_gate_kernel(double LB,double *vx_dev,double *vy_dev,double *dx_dev,double *dy_dev,int *gate_dev)
{
  double r2;  
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
 
 if(i_global<NP){
    dx_dev[i_global]+=vx_dev[i_global]*dt; //particles dispacement update
    dy_dev[i_global]+=vy_dev[i_global]*dt;
    r2 = dx_dev[i_global]*dx_dev[i_global]+dy_dev[i_global]*dy_dev[i_global];
    // printf("dr2=%d\n",dr2);
    if(r2> 0.25*(RCHK-rcut)*(RCHK-rcut)){ //threshold for update. rcut = potential cut distance.
      gate_dev[0]=1; //update's gate open!! many particle rewrite okay!
      //printf("i=%d,dr=%f\n",i_global,sqrt(r2));
    }
  }
}


__global__ void update(double LB,double *x_dev,double *y_dev,double *dx_dev,double *dy_dev,int *list_dev,int *gate_dev)
{
  double dx,dy,r2;  
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(gate_dev[0] == 1 && i_global<NP){
    list_dev[NN*i_global]=0;     // single array in GPU, = list[i][0] 
    for (int j=0; j<NP; j++)
      if(j != i_global){  //ignore self particle
	dx = x_dev[i_global] - x_dev[j];
	dy = y_dev[i_global] - y_dev[j];
	dx -= LB*floor(dx/LB+0.5); //boudary condition
	dy -= LB*floor(dy/LB+0.5);	  
	r2 = dx*dx + dy*dy;
	if(r2 < RCHK*RCHK){
	  list_dev[NN*i_global]++; //list[i][0] ++;
	  list_dev[NN*i_global+list_dev[NN*i_global]]=j; //list[i][list[i][0]] = j;
	}
      }
    //  printf("i=%d, list=%d\n",i_global,list_dev[NN*i_global]);      
    dx_dev[i_global]=0.; //reset dx
    dy_dev[i_global]=0.; //
    if(i_global==0) //only first thread change to the gate value
      gate_dev[0] = 0; // for only single thread.
  }
}


__global__ void calc_force_kernel(double*x_dev,double*y_dev,double *fx_dev,double *fy_dev,double *a_dev,double LB,int *list_dev){
  double dx,dy,dr,dU,a_ij;
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  //a_i  = a_dev[i_global];

  if(i_global<NP){
    fx_dev[i_global] = 0.0;
    fy_dev[i_global] = 0.0;
    for(int j = 1; j<=list_dev[NN*i_global]; j++){ //list[i][0]
      dx=x_dev[list_dev[NN*i_global+j]]-x_dev[i_global]; //x[list[i][j]-x[i]
      dy=y_dev[list_dev[NN*i_global+j]]-y_dev[i_global];
      
      dx -= LB*floor(dx/LB+0.5);
      dy -= LB*floor(dy/LB+0.5);	
      dr = sqrt(dx*dx+dy*dy);
      a_ij=0.5*(a_dev[i_global]+a_dev[list_dev[NN*i_global+j]]);  //0.5*(a[i]+a[i][j])
      if(dr < a_ij){ //cut off
	      dU = -(1-dr/a_ij)/a_ij; //derivertive of U wrt r for harmonic potential.
         fx_dev[i_global] += dU*dx/dr; //only deal for i_global, don't care the for "j"
         fy_dev[i_global] += dU*dy/dr;
      }     
    }
    // printf("i=%d, fx=%f\n",i_global,fx_dev[i_global]);
  }
}

__global__ void calc_force_BHHP_kernel(double*x_dev,double*y_dev,double *fx_dev,double *fy_dev,double *a_dev,double LB,int *list_dev){
  double dx,dy,dU,a_ij,r2, w2,w4,w12,cut;
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  //a_i  = a_dev[i_global];
  cut = 3.0;
  if(i_global<NP){
    fx_dev[i_global] = 0.0;
    fy_dev[i_global] = 0.0;
    for(int j = 1; j<=list_dev[NN*i_global]; j++){ //list[i][0]
      dx= x_dev[list_dev[NN*i_global+j]] - x_dev[i_global]; //x[list[i][j]-x[i]
      dy= y_dev[list_dev[NN*i_global+j]] - y_dev[i_global];
      
      dx -= LB*floor(dx/LB+0.5);
      dy -= LB*floor(dy/LB+0.5);	
      //dr = sqrt(dx*dx+dy*dy);
      a_ij=0.5*(a_dev[i_global]+a_dev[list_dev[NN*i_global+j]]);  //0.5*(a[i]+a[i][j])
      r2 = dx * dx + dy * dy;
      w2 = a_ij * a_ij / r2;
      w4 = w2*w2;
      w12 = w4*w4*w4;
      if(r2 < cut*cut){ //cut off
	      dU = (-12.0)*w12/r2; //derivertive of U wrt r for harmonic potential.
         fx_dev[i_global] += dU*dx; //only deal for i_global, don't care the for "j"
         fy_dev[i_global] += dU*dy;
      }     
    }
    // printf("i=%d, fx=%f\n",i_global,fx_dev[i_global]);
  }
}

__global__ void copy_kernel(double *x0_dev, double *y0_dev, double *x_dev, double *y_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
  x0_dev[i_global]=x_dev[i_global];
  y0_dev[i_global]=y_dev[i_global];
  //printf("%f,%f\n",x_dev[i_global],x0_dev[i_global]);
  } 
}


//For MSD measure, save initial positions
__global__ void copy_kernel2(double *xi_dev, double *yi_dev, double *x_dev, double *y_dev, double *corr_x_dev, double *corr_y_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
    xi_dev[i_global] = x_dev[i_global] - *corr_x_dev;
    yi_dev[i_global] = y_dev[i_global] - *corr_y_dev;
    //if(i_global%1000==0){printf("%d, %f, %f\n",i_global, *corr_x_dev,*corr_y_dev);}
  } 
}


/*
__global__ void init_gate_kernel(int *gate_dev){
  gate_dev[0]=0;
}*/

// homogeneous system -> binary you have to change
__global__ void init_array(double *x_dev, double c){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  x_dev[i_global] = c;
}

__global__ void init_binary(double *x_dev, double c, double c2){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global < NP){ 
    if(i_global%2==0){x_dev[i_global] =c;}
    else {x_dev[i_global] =c2;}
  }
}

//position randomly allocate
__global__ void init_array_rand(double *x_dev, double c,curandState *state){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  x_dev[i_global] = c*curand_uniform(&state[i_global]); 
}


__global__ void MSD_ISF_device(double *x_dev, double *y_dev, double *xi_dev, double *yi_dev, double *corr_x_dev, double *corr_y_dev, double *MSD_dev, double *ISF_dev, double LB){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  double dx, dy;
  double q = 2. * M_PI / 1.0;

  if (i_global<NP){
     dx = x_dev[i_global] - xi_dev[i_global] - *corr_x_dev;
     dy = y_dev[i_global] - yi_dev[i_global] - *corr_y_dev;
    //if(i_global%1000==0){printf("%d  %.3f %.3f\n", i_global, dx, dy);}
     dx -= LB*floor(dx/LB+0.5); //boudary condition
     dy -= LB*floor(dy/LB+0.5);	  
     
     MSD_dev[i_global] = (dx*dx + dy*dy)/(double)NP;
     ISF_dev[i_global] = (cos(- q * dx) + cos(- q * dy)) / (double)NP / 2.0;
     //if(i_global%1000==0){printf("%d	%.4f\n",i_global, *corr_x_dev);}
  }
}

__global__ void ISF_device(double *x_dev, double *y_dev, double *xi_dev, double *yi_dev, double *corr_x_dev, double *corr_y_dev, double *ISF_dev, double LB){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  double dx, dy;
  double q = 2. * M_PI / 1.0;
  if (i_global<NP){
     dx = x_dev[i_global] - xi_dev[i_global] - *corr_x_dev; //including COM displacement 
     dy = y_dev[i_global] - yi_dev[i_global] - *corr_y_dev;

     dx -= LB*floor(dx/LB+0.5); //boudary condition
     dy -= LB*floor(dy/LB+0.5);	  

     ISF_dev[i_global] = 0.5 * (cos(-q * dx) + cos(-q * dy)) / (double)NP;
     //if(i_global%1000==0){printf("%d	%.4f\n",i_global, *corr_x_dev);}
  }
}

void output_Measure(double *measure_time, double *MSD, double *ISF, double *count, int time_count, int eq_count ){
  char filename[128];
  sprintf(filename,"MSD_ISF_T=%.4f.dat",temp);
  FILE *fp;
  fp = fopen(filename, "w+");
 
  for(int i=1;i<time_count;i++){
    fprintf(fp, "%.4f\t%.4f\t%.4f\n", measure_time[i]-measure_time[0], MSD[i]/(count[i]-eq_count), ISF[i]/(count[i]-eq_count));
  }
  fclose(fp);
}

int output(double *x,double *y,double *r1, double t, double dt){
  int i;
  static int count=1;
  char filename[128], foldername[128];
  sprintf(foldername, "position_%.3f/position", temp);
  mkdir(foldername, 0755);
  sprintf(filename,"%s/bina2d_%.d.dat",foldername, count);
  ofstream file;
  file.open(filename);

  for(i=0;i<NP;i++)
    file << x[i] << " " << y[i]<< " " << r1[i] << endl;
  file.close();
  count++;
  return 0;
}

int output_t(double *x,double *y, double t, double time_stamp, double x_corr,double y_corr, double K)
{
  int i;
  char filename[64];
  char filename2[64];
  FILE *fp;
  FILE *fp2;
  char foldername[64];
  sprintf(foldername, "position_%.3f", temp);
  mkdir(foldername, 0755);
  sprintf(filename,"%s/time_coord_%.3f.dat", foldername, temp);
  fp=fopen(filename,"a+");

  for(i=0;i<NP;i++)
    fprintf(fp,"%f\t%f\t%f\n",t-time_stamp, x[i]-x_corr, y[i]-y_corr);
  fclose(fp);

  sprintf(filename2,"%s/time_energy_%.3f.dat", foldername, temp);
  fp2=fopen(filename2,"a+");
  fprintf(fp2,"%f\t%f\t%f\t%f\n", t-time_stamp, x_corr, y_corr, K);
  fclose(fp2);

  return 0;
}

/*
void output(double *x,double *y,double *vx,double *vy,double *a){
  static int count=1;
  char filename[128];
  sprintf(filename,"coord_%.d.dat",count);
  ofstream file;
  file.open(filename);
  double temp0=0.0;
  
  for(int i=0;i<NP;i++){
    file << x[i] << " " << y[i]<< " " << a[i] << endl;
    temp0 += 0.5*(vx[i]*vx[i]+vy[i]*vy[i]);
    //  cout <<vx[i]<<endl;
  }
  file.close();

  cout<<"temp="<< temp0/NP <<endl;
  count++;
}*/


int main(){
  double *x,*xi,*x0_dev,*vx,*y,*yi,*y0_dev,*vy,*a,*x_dev,*xi_dev,*vx_dev,*y_dev,*yi_dev, *dx_dev,*dy_dev,*vy_dev,*a_dev,*fx_dev,*fy_dev; 
  double *x_corr_dev, *y_corr_dev, *x_corr, *y_corr, corr_x=0., corr_y=0., *corr_x_dev, *corr_y_dev;
  int *list_dev,*gate_dev, time_count, init_count;
  double *MSD_dev, *MSD_host, *ISF_dev,*ISF_host;
  double sampling_time, time_stamp=0.;
  double sampling_time_max =1e+4;
  curandState *state; //Cuda state for random numbers
  double sec; //measurred time
  double noise_intensity = sqrt(2.*zeta*temp*dt); //Langevin noise intensity.  
  double LB = sqrt((double)NP/rho);  //system size
 
  x  = (double*)malloc(NB*NT*sizeof(double)); // memory allocattion on CPU
  xi  = (double*)malloc(NB*NT*sizeof(double)); // memory allocattion on CPU
  y  = (double*)malloc(NB*NT*sizeof(double));
  yi  = (double*)malloc(NB*NT*sizeof(double));
  vx = (double*)malloc(NB*NT*sizeof(double));
  vy = (double*)malloc(NB*NT*sizeof(double));
  a  = (double*)malloc(NB*NT*sizeof(double));
  x_corr  = (double*)malloc(NB*NT*sizeof(double));
  y_corr  = (double*)malloc(NB*NT*sizeof(double));
  MSD_host  = (double*)malloc(NB*NT*sizeof(double));
  ISF_host  = (double*)malloc(NB*NT*sizeof(double));
  cudaMalloc((void**)&x_dev,  NB * NT * sizeof(double)); // CudaMalloc should be executed once in the host. 
  cudaMalloc((void**)&y_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&xi_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&yi_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&x0_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&y0_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&dx_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&dy_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&vx_dev, NB * NT * sizeof(double)); 
  cudaMalloc((void**)&vy_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&fx_dev, NB * NT * sizeof(double)); 
  cudaMalloc((void**)&fy_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&x_corr_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&y_corr_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&corr_x_dev, sizeof(double));
  cudaMalloc((void**)&corr_y_dev, sizeof(double));
  cudaMalloc((void**)&MSD_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&ISF_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&a_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&gate_dev, sizeof(int)); // for update 
  cudaMalloc((void**)&list_dev,  NB * NT * NN* sizeof(int)); 
  cudaMalloc((void**)&state,  NB * NT * sizeof(curandState)); 

  sampling_time = 5.*dt;
  time_count = 0;
  
  for(double t=dt;t<timemax;t+=dt){
    if(int(t/dt)== int((sampling_time + time_stamp)/dt)){
	sampling_time *=pow(10,0.1);
	sampling_time=int(sampling_time/dt)*dt;
	time_count++;
	//printf("%.5f	%d\n",t, time_count);
	if(sampling_time > sampling_time_max/pow(10.,0.1)){
	  time_stamp=0.;
	  sampling_time=5.*dt;
	  break;
      }
    }
  } 
  

int max_count = time_count;
double measure_time[time_count], MSD[time_count], count[time_count], ISF[time_count];
  //Make the measure time table
  time_count = 0.;
  for(double t=dt;t<timemax;t+=dt){
    if(int(t/dt)== int((sampling_time + time_stamp)/dt)){
      count[time_count] = 0.;
      MSD[time_count] = 0.;
      ISF[time_count] = 0.;
    	measure_time[time_count] = t - time_stamp;
	sampling_time *=pow(10,0.1);
	sampling_time=int(sampling_time/dt)*dt;
	printf("%.5f	%d\n", measure_time[time_count], time_count);
	time_count++;
	if(sampling_time > sampling_time_max/pow(10.,0.1)){
	  time_stamp=0.;//reset the time stamp
	  sampling_time=5.*dt; //reset the time sampling_time
	  break;
      }
    }
  }
  
  setCurand<<<NB,NT>>>(rand(), state); // Construction of the cudarand state.  
  init_array_rand<<<NB,NT>>>(x_dev,LB,state);
  init_array_rand<<<NB,NT>>>(y_dev,LB,state);
  init_binary<<<NB,NT>>>(a_dev,1.0, 1.4);
  init_array<<<NB,NT>>>(vx_dev,0.);
  init_array<<<NB,NT>>>(vy_dev,0.);
  init_array<<<NB,NT>>>(x_corr_dev,0.);
  init_array<<<NB,NT>>>(y_corr_dev,0.);
  init_array<<<NB,NT>>>(ISF_dev,0.);
  init_array<<<NB,NT>>>(MSD_dev,0.);
  ini_gate_kernel<<<1,1>>>(gate_dev,1);//block variable
  cudaMemcpy(a, a_dev,  NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  update<<<NB,NT>>>(LB,x_dev,y_dev,dx_dev,dy_dev,list_dev,gate_dev); 
 
  for(double t=0;t<timeeq;t+=dt){
    // cout<<t<<endl;
    calc_force_kernel<<<NB,NT>>>(x_dev,y_dev,fx_dev,fy_dev,a_dev,LB,list_dev);
    langevin_kernel<<<NB,NT>>>(x_dev,y_dev,vx_dev,vy_dev,fx_dev,fy_dev,state,0.0,LB);
    disp_gate_kernel<<<NB,NT>>>(LB,vx_dev,vy_dev,dx_dev,dy_dev,gate_dev); //for auto-list method
    update<<<NB,NT>>>(LB,x_dev,y_dev,dx_dev,dy_dev,list_dev,gate_dev);
  }

  measureTime();
  time_count = 0;
  init_count = 0;
  int eq_count = 2;
  
  for(double t=dt;t<timemax;t+=dt){
    calc_force_BHHP_kernel<<<NB,NT>>>(x_dev,y_dev,fx_dev,fy_dev,a_dev,LB,list_dev);
    langevin_kernel<<<NB,NT>>>(x_dev,y_dev,vx_dev,vy_dev,fx_dev,fy_dev,state,noise_intensity,LB);
    //cudaMemcpy(x, x_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
    //cudaMemcpy(y, y_dev,  NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
    com_correction<<<NB,NT>>>(x_dev, y_dev, x_corr_dev, y_corr_dev, LB);
    //com_correction(x, y, &corr_x, &corr_y, LB);
    //cudaDeviceSynchronize();
    //p_bound<<<NB,NT>>>(x_dev,y_dev,LB);
    //RDF, Sq measure
    if(int(t/dt)== int((sampling_time + time_stamp)/dt)){
	    count[time_count]++;//measure count at each logarithmic times
      cudaMemcpy(x_corr, x_corr_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
      cudaMemcpy(y_corr, y_corr_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
      calc_com(x_corr, y_corr, &corr_x, &corr_y);
      cudaMemcpy(corr_x_dev, &corr_x, sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(corr_y_dev, &corr_y, sizeof(double),cudaMemcpyHostToDevice);
      //cudaDeviceSynchronize();
      if(init_count>=eq_count){
        if(time_count==0){
          //cudaDeviceSynchronize();
          copy_kernel2<<<NB,NT>>>(xi_dev, yi_dev, x_dev,y_dev, corr_x_dev, corr_y_dev);
        }
        MSD_ISF_device<<<NB,NT>>>(x_dev, y_dev, xi_dev, yi_dev, corr_x_dev, corr_y_dev, MSD_dev, ISF_dev, LB);
        //ISF_device<<<NB,NT>>>(x_dev, y_dev, xi_dev, yi_dev, corr_x_dev, corr_y_dev, ISF_dev, LB);
        cudaMemcpy(MSD_host, MSD_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
        cudaMemcpy(ISF_host, ISF_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);

        double MSD_temp = calc_MSD(MSD_host); //the variable for check in real-time
        double ISF_temp = calc_ISF(ISF_host); //If you don't need to check, using just sub-routines
        
        MSD[time_count] += MSD_temp;//reduce the MSD from each particles
        ISF[time_count] += ISF_temp; //reduce the ISF from each particles

        printf("%d %d	%.4f	%.4f  %.4f  %.4f  %.4f\n", time_count, init_count, measure_time[time_count], MSD_temp, ISF_temp, corr_x, corr_y);
        //cudaMemcpy(vx, vx_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
        //cudaMemcpy(vy, vy_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
        //double K = calc_K(vx,vy);
        //output_t(x, y, t, time_stamp, corr_x, corr_y, K);

      }

      else {printf("time = %.4f\n", measure_time[time_count]);}
	    sampling_time *=pow(10,0.1);
	    sampling_time=int(sampling_time/dt)*dt;
	    time_count++;
      
	    if(sampling_time > sampling_time_max/pow(10.,0.1)){
	      time_stamp=t; //memory of initial measure time for logarithmic sampling
	      sampling_time=5.*dt; //reset the time sampling_time
	      init_count++;
        time_count = 0;
      }
    }

    disp_gate_kernel<<<NB,NT>>>(LB,vx_dev,vy_dev,dx_dev,dy_dev,gate_dev); //for auto-list method
    // cudaDeviceSynchronize(); // for printf in the device.
    update<<<NB,NT>>>(LB,x_dev,y_dev,dx_dev,dy_dev,list_dev,gate_dev);
    // cudaDeviceSynchronize();
    // cout <<t<<endl;
  } 

  sec = measureTime()/1000.;
  cout<<"time(sec):"<<sec<<endl;
  cudaMemcpy(x,   x_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(vx, vx_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(y,   y_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(vy, vy_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(a, a_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  

  //output(x,y,vx,vy,a);
  output_Measure(measure_time, MSD, ISF, count, max_count, eq_count);
  cudaFree(x_dev);
  cudaFree(xi_dev);
  cudaFree(vx_dev);
  cudaFree(yi_dev);
  cudaFree(vy_dev);
  cudaFree(dx_dev);
  cudaFree(dy_dev);
  cudaFree(x_corr_dev);
  cudaFree(y_corr_dev);
  cudaFree(corr_x_dev);
  cudaFree(corr_y_dev);
  cudaFree(MSD_dev);
  cudaFree(ISF_dev);
  cudaFree(gate_dev);
  cudaFree(state);
  cudaFree(a_dev);
  free(x); 
  free(xi); 
  free(vx); 
  free(y); 
  free(yi); 
  free(vy); 
  free(x_corr); 
  free(y_corr); 
  free(MSD_host); 
  free(ISF_host);
  free(a); 
  return 0;
}
