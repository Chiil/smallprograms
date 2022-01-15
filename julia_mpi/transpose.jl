## User input.
npx = 4; npy = 4
itot = 8; jtot = 8; ktot = 8

imax = itot ÷ npx; jmax = jtot ÷ npy; kmax = ktot ÷ npx


## Init MPI and create grid.
using MPI

MPI.Init()

dims = [npx, npy]; periodic = [1, 1]; reorder = true
commxy = MPI.Cart_create(MPI.COMM_WORLD, dims, periodic, reorder)
mpiid = MPI.Comm_rank(commxy)
commx = MPI.Cart_sub(commxy, [false, true])

data = ones(Int, imax, jmax, ktot) * mpiid
data_new = Array{Int}(undef, itot, jmax, kmax)

sendbuf = Array{Int}(undef, imax, jmax, kmax, npx)
recvbuf = Array{Int}(undef, imax, jmax, kmax, npx)


## Load the buffer
for i in 1:npx
    ks = (i-1)*kmax + 1
    ke = i*kmax
    sendbuf[:, :, :, i] .= data[:, :, ks:ke]
end

print("s $mpiid: $sendbuf \n")


## Transpose zx
# reqs = Vector{MPI.Request}(undef, 2*npx)
# for i in 1:npx
#     tag = 1
#     reqs[2*(i-1)+1] = MPI.Irecv!(recvbuf[:, :, :, i], i-1, tag, commx)
#     reqs[2*(i-1)+2] = MPI.Isend( sendbuf[:, :, :, i], i-1, tag, commx)
# end
# MPI.Waitall!(reqs)
MPI.Alltoall!(MPI.UBuffer(sendbuf, imax*jmax*kmax), MPI.UBuffer(recvbuf, imax*jmax*kmax), commx)

print("r $mpiid: $recvbuf \n")


## Unload the buffer
for i in 1:npx
    is = (i-1)*imax + 1
    ie = i*imax
    data_new[is:ie, :, :] .= recvbuf[:, :, :, i]
end

# print("d $mpiid: $data_new \n")
