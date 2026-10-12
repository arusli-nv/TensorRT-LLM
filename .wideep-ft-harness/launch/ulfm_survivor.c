#include <mpi.h>
#include <mpi-ext.h>
#include <signal.h>
#include <stdio.h>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size, one = 1, sum = 0, cls = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    MPI_Allreduce(&one, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == size - 1)
    {
        raise(SIGKILL);
    }
    int rc = MPI_Allreduce(&one, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Error_class(rc, &cls);
    MPIX_Comm_revoke(MPI_COMM_WORLD);
    MPI_Comm survivors;
    MPIX_Comm_shrink(MPI_COMM_WORLD, &survivors);
    int ssize;
    MPI_Comm_size(survivors, &ssize);
    sum = 0;
    rc = MPI_Allreduce(&one, &sum, 1, MPI_INT, MPI_SUM, survivors);
    printf("rank %d proc_failed=%d shrunk=%d survivor_sum=%d rc=%d\n", rank,
        cls == MPIX_ERR_PROC_FAILED || cls == MPIX_ERR_REVOKED, ssize, sum, rc);
    fflush(stdout);
    MPI_Comm_free(&survivors);
    MPI_Finalize();
    return 0;
}
