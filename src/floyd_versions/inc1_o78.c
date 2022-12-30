#include <floyd_warshal.h> //It implements this header file.
#include "../src/floyd_versions/common/opt_0-n.c"
#include "../src/floyd_versions/common/malloc/aligned.c"

#define likely(x)   __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

#define UNROLL_FACTOR BS/(SIMD_WIDTH/TYPE_SIZE)

// ----------------- BLOQUE AGREGADO -------------------

#include <pthread.h>

// ---------------- FIN BLOQUE AGREGADO --------------

//Public
char* getFloydName(){
	return "cond variables with loop unroll";
}

//Public
char* getFloydVersion(){
	return "inc1 Opt-7_8";
}

static inline void FW_BLOCK(TYPE* const graph, const INT64 d1, const INT64 d2, const INT64 d3, int* const path, const INT64 base, int* const tmp1, int* const tmp2) __attribute__((always_inline));

//Private
#pragma GCC diagnostic ignored "-Wunused-parameter"
static inline void FW_BLOCK(TYPE* const graph, const INT64 d1, const INT64 d2, const INT64 d3, int* const path, const INT64 base, int* const tmp1, int* const tmp2){
	INT64 i, j, k, i_disp, i_disp_d1, k_disp, k_disp_d3;
	TYPE dij, dik, dkj, sum;

	for(k=0; k<BS; k++){
		k_disp = k*BS;
		k_disp_d3 = k_disp + d3;
		for(i=0; i<BS; i+=2){
			i_disp = i*BS;
			i_disp_d1 = i_disp + d1;
			dik = graph[i_disp + d2 + k];		
			#ifdef INTEL_ARC
				#pragma unroll(UNROLL_FACTOR)
			#endif
			#pragma omp simd private(dij,dkj,sum)
			for(j=0; j<BS; j++){
				dij = graph[i_disp_d1 + j];
				dkj = graph[k_disp_d3 + j];
				sum = dik + dkj;
				if(unlikely(sum < dij)){
					graph[i_disp_d1 + j] = sum;
					#ifndef NO_PATH
						path[i_disp_d1 + j] = base + k;
					#else
						tmp1[j] = tmp2[j];
					#endif
				}
			}
			i_disp = (i+1)*BS;
			i_disp_d1 = i_disp + d1;
			dik = graph[i_disp + d2 + k];			
			#ifdef INTEL_ARC
				#pragma unroll(UNROLL_FACTOR)
			#endif
			#pragma omp simd private(dij,dkj,sum)
			for(j=0; j<BS; j++){
				dij = graph[i_disp_d1 + j];
				dkj = graph[k_disp_d3 + j];
				sum = dik + dkj;
				if(unlikely(sum < dij)){
					graph[i_disp_d1 + j] = sum;
					#ifndef NO_PATH
						path[i_disp_d1 + j] = base + k;
					#else
						tmp1[j] = tmp2[j];
					#endif
				}
			}
		}
	}
}

static inline void FW_BLOCK_PARALLEL(TYPE* const graph, const INT64 d1, const INT64 d2, const INT64 d3, int* const path, const INT64 base, int* const tmp1, int* const tmp2) __attribute__((always_inline));

//Private
#pragma GCC diagnostic ignored "-Wunused-parameter"
static inline void FW_BLOCK_PARALLEL(TYPE* const graph, const INT64 d1, const INT64 d2, const INT64 d3, int* const path, const INT64 base, int* const tmp1, int* const tmp2){
	INT64 i, j, k, i_disp, i_disp_d1, k_disp, k_disp_d3;
	TYPE dij, dik, dkj, sum;

	for(k=0; k<BS; k++){
		k_disp = k*BS;
		k_disp_d3 = k_disp + d3;
		#pragma omp for
		for(i=0; i<BS; i+=2){
			i_disp = i*BS;
			i_disp_d1 = i_disp + d1;
			dik = graph[i_disp + d2 + k];
			#ifdef INTEL_ARC
				#pragma unroll(UNROLL_FACTOR)
			#endif  
			#pragma omp simd private(dij,dkj,sum)
			for(j=0; j<BS; j++){
				dij = graph[i_disp_d1 + j];
				dkj = graph[k_disp_d3 + j];
				sum = dik + dkj;
				if(unlikely(sum < dij)){
					graph[i_disp_d1 + j] = sum;
					#ifndef NO_PATH
						path[i_disp_d1 + j] = base + k;
					#else
						tmp1[j] = tmp2[j];
					#endif
				}
			}
			i_disp = (i+1)*BS;
			i_disp_d1 = i_disp + d1;
			dik = graph[i_disp + d2 + k];
			#ifdef INTEL_ARC
				#pragma unroll(UNROLL_FACTOR)
			#endif  
			#pragma omp simd private(dij,dkj,sum)
			for(j=0; j<BS; j++){
				dij = graph[i_disp_d1 + j];
				dkj = graph[k_disp_d3 + j];
				sum = dik + dkj;
				if(unlikely(sum < dij)){
					graph[i_disp_d1 + j] = sum;
					#ifndef NO_PATH
						path[i_disp_d1 + j] = base + k;
					#else
						tmp1[j] = tmp2[j];
					#endif
				}
			}
		}
	}
}

//Public
void floydWarshall(TYPE* D, int* P, int n, int t){
	INT64 r, row_of_blocks_disp, num_of_bock_elems;	
	r = n/BS;
	row_of_blocks_disp = n*BS;
	num_of_bock_elems = BS*BS;
	
	// --------------------------- BLOQUE AGREGADO -----------------------

	INT64 x, y;
	char** pendientes;		// no supera el valor 255
	pthread_cond_t** cv;
	pthread_mutex_t** mutex;

	// asignación de memoria (convenientemente matrices separadas)
	pendientes = (char**) malloc(r * sizeof(char*));
	for (x=0; x<r; x++) pendientes[x] = (char*) malloc(r * sizeof(char));

	cv = (pthread_cond_t**) malloc(r * sizeof(pthread_cond_t*));
	for (x=0; x<r; x++) cv[x] = (pthread_cond_t*) malloc(r * sizeof(pthread_cond_t));

	mutex = (pthread_mutex_t**) malloc(r * sizeof(pthread_mutex_t*));
	for (x=0; x<r; x++) mutex[x] = (pthread_mutex_t*) malloc(r * sizeof(pthread_mutex_t));

	// inicialización de pendientes
	for (x=0; x<r; x++){
		for (y=0; y<r; y++){
			pthread_cond_init(&cv[x][y], NULL);
			pthread_mutex_init(&mutex[x][y], NULL);
		}
	}

	// ------------------------- FIN BLOQUE AGREGADO -----------------------
	
	// Modificación: shared(pendientes, cv, mutex)
	
	#pragma omp parallel shared(pendientes, cv, mutex) default(none) firstprivate(r,row_of_blocks_disp,num_of_bock_elems,D,P) num_threads(t)
	{
		INT64 i, j, k, b, kj, ik, kk, ij, k_row_disp, k_col_disp, i_row_disp, j_col_disp, w;
		
		// Variable agregada
		INT64 aux;
		
		#ifndef NO_PATH
			int* tmp1 = NULL;
			int* tmp2 = NULL;
		#else
			int tmp1[BS], tmp2[BS];
			//tmp1 and tmp2 are used as a patch to avoid a compiler bug which makes it lose performance instead of winning while omitting the compute of the P patrix
		#endif
		
		for(k=0; k<r; k++){
			b = k*BS;
			k_row_disp = k*row_of_blocks_disp;
			k_col_disp = k*num_of_bock_elems;

			//Phase 1
			kk = k_row_disp + k_col_disp;
			FW_BLOCK_PARALLEL(D, kk, kk, kk, P, b, tmp1, tmp2);

			// -------------------- BLOQUE AGREGADO ------------------------
			
			// Asignación de pendientes
			#pragma omp for collapse(2) schedule(dynamic)
			for (i=0; i<r; i++){
				for (j=0; j<r; j++){
					pendientes[i][j] = 2;
				}
			}
			
			// ------------------- FIN BLOQUE AGREGADO -----------------------

			//Phase 2 y 3
			#pragma omp for schedule(dynamic)
			for(w=0; w<r*2; w++){
				if(w<r){ 
					//Phase 2
					j = w;
					if(j == k) continue;
					kj = k_row_disp + j*num_of_bock_elems;
					FW_BLOCK(D, kj, kk, kj, P, b, tmp1, tmp2);
					
					// -------------- BLOQUE AGREGADO -------------------

					// Finalizó el computo del bloque (k,j) = (k,w)
					// Modif: se debe decrementar pendientes de la columna actual "j"
					for (aux=0; aux<r; aux++){
						if (aux == k) continue;					// se ignora actual (k,j)
						pthread_mutex_lock(&mutex[aux][j]);
						pendientes[aux][j]--;
						pthread_cond_signal(&cv[aux][j]);
						pthread_mutex_unlock(&mutex[aux][j]);
					}

					// -------------- FIN BLOQUE AGREGADO -----------------
					
				} else { 
					//Phase 3
					i = w - r;
					if(i == k) continue;
					ik = i*row_of_blocks_disp + k_col_disp;
					FW_BLOCK(D, ik, ik, kk, P, b, tmp1, tmp2);
					
					// -------------- BLOQUE AGREGADO -------------------

					// Finalizo el computo del bloque (i,k) = (w-r, k)
					// Modif: se debe decrementar pendientes de la fila actual "i"
					for (aux=0; aux<r; aux++) {
						if (aux == k) continue;					// se ignora actual (i,k)
						pthread_mutex_lock(&mutex[i][aux]);
						pendientes[i][aux]--;
						pthread_cond_signal(&cv[i][aux]);
						pthread_mutex_unlock(&mutex[i][aux]);
					}

					// -------------- FIN BLOQUE AGREGADO -----------------
					
				}
			}

			//Phase 4
			#pragma omp for collapse(2) schedule(dynamic)
			for(i=0; i<r; i++){
				for(j=0; j<r; j++){
					if( (j == k) || (i == k) ) continue;
					
					// ----------- BLOQUE AGREGADO -----------------

					// Esperar que se computen los bloques (k,j) e (i,k)
					pthread_mutex_lock(&mutex[i][j]);
					while (pendientes[i][j] > 0) pthread_cond_wait(&cv[i][j], &mutex[i][j]);
					pthread_mutex_unlock(&mutex[i][j]);

					// ---------- FIN BLOQUE AGREGADO --------------
					
					i_row_disp = i*row_of_blocks_disp;
					ik = i_row_disp + k_col_disp;
					j_col_disp = j*num_of_bock_elems;
					kj = k_row_disp + j_col_disp;
					ij = i_row_disp + j_col_disp;
					FW_BLOCK(D, ij, ik, kj, P, b, tmp1, tmp2);
				}
			}
		}
	}
}
