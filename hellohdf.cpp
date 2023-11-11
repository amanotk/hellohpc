#include <hdf5.h>
#include <iostream>
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

bool error_check(int status, const char* message)
{
  if (status < 0) {
    std::cerr << message << std::endl;
    return false;
  }
  return true;
}

void fill_data(int data[], const int nx, const int ny)
{
  int thisrank  = 0;
  int nprocess  = 0;
  int nx_global = 0;
  int ix_start  = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocess);

  nx_global = nx * nprocess;
  ix_start  = nx * thisrank;

  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      data[iy * nx + ix] = iy * nx_global + ix + ix_start;
    }
  }
}

bool check_data(int data1[], int data2[], const int nx, const int ny)
{
  bool status = true;

  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      status = status && (data1[iy * nx + ix] == data2[iy * nx + ix]);
    }
  }

  return status;
}

bool check_create_file(const char* filename)
{
  bool   is_success = true;
  herr_t status;

  hid_t property = H5Pcreate(H5P_FILE_ACCESS);
  is_success &= error_check(property, "Error creating property list.");

  status = H5Pset_fapl_mpio(property, MPI_COMM_WORLD, MPI_INFO_NULL);
  is_success &= error_check(status, "Error setting file access property list.");

  hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, property);
  is_success &= error_check(file, "Error creating file.");

  status = H5Fclose(file);
  is_success &= error_check(status, "Error closing file.");

  status = H5Pclose(property);
  is_success &= error_check(status, "Error closing property list.");

  return is_success;
}

bool check_create_dataset(const char* filename, const char* name, hid_t type, const int ndim,
                          const hsize_t dims[])
{
  bool   is_success = true;
  herr_t status;

  hid_t property = H5Pcreate(H5P_FILE_ACCESS);
  is_success &= error_check(property, "Error creating property list.");

  status = H5Pset_fapl_mpio(property, MPI_COMM_WORLD, MPI_INFO_NULL);
  is_success &= error_check(status, "Error setting file access property list.");

  hid_t file = H5Fopen(filename, H5F_ACC_RDWR, property);
  is_success &= error_check(file, "Error opening file.");

  hid_t space = H5Screate_simple(ndim, dims, NULL);
  is_success &= error_check(space, "Error creating dataspace.");

  hid_t dataset = H5Dcreate(file, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  is_success &= error_check(dataset, "Error creating dataset.");

  status = H5Dclose(dataset);
  is_success &= error_check(status, "Error closing dataset.");

  status = H5Sclose(space);
  is_success &= error_check(status, "Error closing dataspace.");

  status = H5Fclose(file);
  is_success &= error_check(status, "Error closing file.");

  status = H5Pclose(property);
  is_success &= error_check(status, "Error closing property list.");
  return is_success;
}

bool check_write_dataset(const char* filename, const char* name, const int ndim,
                         const hsize_t ldims[], const hsize_t loffset[], const hsize_t gdims[],
                         const hsize_t goffset[], const void* data)
{
  bool   is_success = true;
  herr_t status;

  hid_t property = H5Pcreate(H5P_FILE_ACCESS);
  is_success &= error_check(property, "Error creating property list.");

  status = H5Pset_fapl_mpio(property, MPI_COMM_WORLD, MPI_INFO_NULL);
  is_success &= error_check(status, "Error setting file access property list.");

  hid_t file = H5Fopen(filename, H5F_ACC_RDWR, property);
  is_success &= error_check(file, "Error opening file.");

  hid_t dataset = H5Dopen(file, name, H5P_DEFAULT);
  is_success &= error_check(dataset, "Error opening dataset.");

  hid_t dspace = H5Dget_space(dataset);
  is_success &= error_check(dspace, "Error getting dataspace.");

  hid_t mspace = H5Screate_simple(ndim, ldims, NULL);
  is_success &= error_check(mspace, "Error creating simple dataspace.");

  hid_t dtype = H5Dget_type(dataset);
  is_success &= error_check(dtype, "Error getting dataset type.");

  hid_t proplist = H5Pcreate(H5P_DATASET_XFER);
  is_success &= error_check(proplist, "Error creating dataset transfer property list.");

  status = H5Sselect_hyperslab(dspace, H5S_SELECT_SET, goffset, NULL, ldims, NULL);
  is_success &= error_check(status, "Error selecting hyperslab for destination.");

  status = H5Sselect_hyperslab(mspace, H5S_SELECT_SET, loffset, NULL, ldims, NULL);
  is_success &= error_check(status, "Error selecting hyperslab for source.");

  status = H5Pset_dxpl_mpio(proplist, H5FD_MPIO_COLLECTIVE);
  is_success &=
      error_check(status, "Error setting dataset transfer property list for collective I/O.");

  status = H5Dwrite(dataset, dtype, mspace, dspace, proplist, data);
  is_success &= error_check(status, "Error writing data.");

  status = H5Pclose(proplist);
  is_success &= error_check(status, "Error closing dataset transfer property list.");

  status = H5Tclose(dtype);
  is_success &= error_check(status, "Error closing datatype.");

  status = H5Sclose(mspace);
  is_success &= error_check(status, "Error closing memory dataspace.");

  status = H5Sclose(dspace);
  is_success &= error_check(status, "Error closing dataspace.");

  status = H5Dclose(dataset);
  is_success &= error_check(status, "Error closing dataset.");

  status = H5Fclose(file);
  is_success &= error_check(status, "Error closing file.");

  status = H5Pclose(property);
  is_success &= error_check(status, "Error closing property list.");

  return is_success;
}

bool check_read_dataset(const char* filename, const char* name, const int ndim,
                        const hsize_t ldims[], const hsize_t loffset[], const hsize_t gdims[],
                        const hsize_t goffset[], void* data)
{
  bool   is_success = true;
  herr_t status;

  hid_t property = H5Pcreate(H5P_FILE_ACCESS);
  is_success &= error_check(property, "Error creating property list.");

  status = H5Pset_fapl_mpio(property, MPI_COMM_WORLD, MPI_INFO_NULL);
  is_success &= error_check(status, "Error setting file access property list.");

  hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, property);
  is_success &= error_check(file, "Error opening file.");

  hid_t dataset = H5Dopen(file, name, H5P_DEFAULT);
  is_success &= error_check(dataset, "Error opening dataset.");

  hid_t dspace = H5Dget_space(dataset);
  is_success &= error_check(dspace, "Error getting dataspace.");

  hid_t mspace = H5Screate_simple(ndim, ldims, NULL);
  is_success &= error_check(mspace, "Error creating simple dataspace.");

  hid_t dtype = H5Dget_type(dataset);
  is_success &= error_check(dtype, "Error getting dataset type.");

  hid_t proplist = H5Pcreate(H5P_DATASET_XFER);
  is_success &= error_check(proplist, "Error creating dataset transfer property list.");

  status = H5Sselect_hyperslab(dspace, H5S_SELECT_SET, goffset, NULL, ldims, NULL);
  is_success &= error_check(status, "Error selecting hyperslab for destination.");

  status = H5Sselect_hyperslab(mspace, H5S_SELECT_SET, loffset, NULL, ldims, NULL);
  is_success &= error_check(status, "Error selecting hyperslab for source.");

  status = H5Pset_dxpl_mpio(proplist, H5FD_MPIO_COLLECTIVE);
  is_success &=
      error_check(status, "Error setting dataset transfer property list for collective I/O.");

  status = H5Dread(dataset, dtype, mspace, dspace, proplist, data);
  is_success &= error_check(status, "Error reading data.");

  status = H5Pclose(proplist);
  is_success &= error_check(status, "Error closing dataset transfer property list.");

  status = H5Tclose(dtype);
  is_success &= error_check(status, "Error closing datatype.");

  status = H5Sclose(mspace);
  is_success &= error_check(status, "Error closing memory dataspace.");

  status = H5Sclose(dspace);
  is_success &= error_check(status, "Error closing dataspace.");

  status = H5Dclose(dataset);
  is_success &= error_check(status, "Error closing dataset.");

  status = H5Fclose(file);
  is_success &= error_check(status, "Error closing file.");

  status = H5Pclose(property);
  is_success &= error_check(status, "Error closing property list.");

  return is_success;
}

int main(int argc, char** argv)
{
  const char* filename = "hellohdf.h5";
  const char* dataname = "data";
  const hid_t datatype = H5T_NATIVE_INT;
  const int   ndim     = 2;
  const int   nx       = 4;
  const int   ny       = 4;

  int thisrank = 0;
  int nprocess = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocess);

  {
    hsize_t ldims[ndim];
    hsize_t gdims[ndim];
    hsize_t loffset[ndim];
    hsize_t goffset[ndim];

    // prepare data
    int data1[ny * nx], data2[ny * nx];
    fill_data(data1, nx, ny);

    ldims[0]   = static_cast<hsize_t>(ny);
    ldims[1]   = static_cast<hsize_t>(nx);
    gdims[0]   = static_cast<hsize_t>(ny);
    gdims[1]   = static_cast<hsize_t>(nx * nprocess);
    loffset[0] = 0;
    loffset[1] = 0;
    goffset[0] = 0;
    goffset[1] = static_cast<hsize_t>(nx * thisrank);

    // create file
    print_message("creating file %s ... ", filename);
    bool create_file_sucess = check_create_file(filename);
    if (create_file_sucess == true) {
      print_message("done\n");
    } else {
      print_message("failed\n");
    }

    // create dataset
    print_message("creating dataset %s ... ", dataname);
    bool create_dataset_success = check_create_dataset(filename, dataname, datatype, ndim, gdims);
    if (create_dataset_success == true) {
      print_message("done\n");
    } else {
      print_message("failed\n");
    }

    // write dataset
    print_message("writing dataset %s ... ", dataname);
    bool write_dataset_sucess =
        check_write_dataset(filename, dataname, ndim, ldims, loffset, gdims, goffset, data1);
    if (write_dataset_sucess == true) {
      print_message("done\n");
    } else {
      print_message("failed\n");
    }

    // read dataset
    print_message("reading dataset %s ... ", dataname);
    bool read_dataset_sucess =
        check_read_dataset(filename, dataname, ndim, ldims, loffset, gdims, goffset, data2);
    if (read_dataset_sucess == true) {
      print_message("done\n");
      print_message("checking data ... ");
      if (check_data(data1, data2, nx, ny) == true) {
        print_message("succeeded\n");
      } else {
        print_message("failed\n");
      }
    } else {
      print_message("failed\n");
    }
  }

  MPI_Finalize();

  return 0;
}