#include <cassert>
#include <iostream>
#include <memory>
#include <mpi.h>

void print_message(const char* format, ...)
{
  int thisrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);
  if (thisrank == 0) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
  }
}

int get_next_rank(int thisrank, int nprocess)
{
  return (thisrank + 1) % nprocess;
}

int get_prev_rank(int thisrank, int nprocess)
{
  return (thisrank - 1 + nprocess) % nprocess;
}

std::unique_ptr<int[]> create_buffer(int size, int fillvalue)
{
  std::unique_ptr<int[]> buffer = std::make_unique<int[]>(size);
  for (int i = 0; i < size; i++) {
    buffer[i] = fillvalue;
  }
  return buffer;
}

void test_mpi_sendrecv(int* sendbuf1, int* sendbuf2, int* recvbuf1, int* recvbuf2, int size,
                       int thisrank, int nprocess)
{
  MPI_Request requests[4];
  int         nextrank = get_next_rank(thisrank, nprocess);
  int         prevrank = get_prev_rank(thisrank, nprocess);

  MPI_Isend(sendbuf1, size, MPI_INT, nextrank, thisrank, MPI_COMM_WORLD, &requests[0]);
  MPI_Isend(sendbuf2, size, MPI_INT, prevrank, thisrank, MPI_COMM_WORLD, &requests[1]);
  MPI_Irecv(recvbuf1, size, MPI_INT, nextrank, nextrank, MPI_COMM_WORLD, &requests[2]);
  MPI_Irecv(recvbuf2, size, MPI_INT, prevrank, prevrank, MPI_COMM_WORLD, &requests[3]);
  MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int thisrank = 0;
  int nprocess = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocess);

  print_message("MPI number of processes %4d\n", nprocess);

  {
    int                    size     = 4;
    std::unique_ptr<int[]> sendbuf1 = create_buffer(size, +thisrank);
    std::unique_ptr<int[]> sendbuf2 = create_buffer(size, -thisrank);
    std::unique_ptr<int[]> recvbuf1 = create_buffer(size, 0);
    std::unique_ptr<int[]> recvbuf2 = create_buffer(size, 0);

    print_message("Testing non-blocking sendrecv ... ");
    test_mpi_sendrecv(sendbuf1.get(), sendbuf2.get(), recvbuf1.get(), recvbuf2.get(), size,
                      thisrank, nprocess);
    print_message("done\n");

    // check the result
    print_message("Checking the result ... ");
    for (int i = 0; i < size; i++) {
      assert(recvbuf1[i] == -get_next_rank(thisrank, nprocess));
      assert(recvbuf2[i] == +get_prev_rank(thisrank, nprocess));
    }
    print_message("succeeded\n");
  }

  MPI_Finalize();

  return 0;
}
