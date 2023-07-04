import subprocess

tile_i = [ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 ]
tile_j = [ 1, 2, 4, 8, 16, 32 ]
tile_k = [ 1, 2, 4 ]

nloop = 3

for k in tile_k:
    for j in tile_j:
        for i in tile_i:
            # Skip illegal and undercapacity
            if i*j*k > 1024 or i*j*k < 32:
                continue
            # compile_string = 'nvc++ -acc -O3 -fast diff_acc.cxx -gpu=cc86 -o diff_acc -DTILE_I={:} -DTILE_J={:} -DTILE_K={:}'.format(i, j, k)
            compile_string = 'nvc++ -acc -O3 -fast advec_2i5_cuda.cu -gpu=cc86 -o advec_cuda -DTILE_I={:} -DTILE_J={:} -DTILE_K={:}'.format(i, j, k)
            subprocess.call(compile_string, shell=True)

            compile_string = 'nvc++ -acc -O3 -fast advec_2i5_acc.cpp -gpu=cc86 -o advec_acc -DTILE_I={:} -DTILE_J={:} -DTILE_K={:}'.format(i, j, k)
            subprocess.call(compile_string, shell=True)

            print(i, j, k, 'CUDA')
            for _ in range(nloop):
                subprocess.call('./advec_cuda 512', shell=True)

            print(i, j, k, 'OpenACC')
            for _ in range(nloop):
                subprocess.call('./advec_acc 512', shell=True)

