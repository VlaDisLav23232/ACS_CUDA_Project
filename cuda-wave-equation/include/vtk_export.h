#pragma once
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace VTK {

inline void write_vti(const std::string& filename,
                      const std::vector<float>& data,
                      int nx, int ny, int nz,
                      float dx = 1.0f, float dy = 1.0f, float dz = 1.0f,
                      const std::string& scalar_name = "displacement") {
    std::ofstream file(filename);
    size_t n_points = static_cast<size_t>(nx) * ny * nz;

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    file << "  <ImageData WholeExtent=\"0 " << (nx - 1) << " 0 " << (ny - 1) << " 0 " << (nz - 1) << "\" ";
    file << "Origin=\"0 0 0\" Spacing=\"" << dx << " " << dy << " " << dz << "\">\n";
    file << "    <Piece Extent=\"0 " << (nx - 1) << " 0 " << (ny - 1) << " 0 " << (nz - 1) << "\">\n";
    file << "      <PointData Scalars=\"" << scalar_name << "\">\n";
    file << "        <DataArray type=\"Float32\" Name=\"" << scalar_name << "\" format=\"ascii\">\n";

    for (size_t i = 0; i < n_points; i++) {
        file << data[i];
        if ((i + 1) % 6 == 0) file << "\n";
        else file << " ";
    }

    file << "\n        </DataArray>\n";
    file << "      </PointData>\n";
    file << "    </Piece>\n";
    file << "  </ImageData>\n";
    file << "</VTKFile>\n";
    file.close();
}

inline void write_timestep_vti(const std::string& base_name,
                               int timestep,
                               const std::vector<float>& data,
                               int nx, int ny, int nz,
                               float dx = 1.0f, float dy = 1.0f, float dz = 1.0f) {
    char filename[256];
    std::snprintf(filename, sizeof(filename), "%s_%04d.vti", base_name.c_str(), timestep);
    write_vti(filename, data, nx, ny, nz, dx, dy, dz);
}

inline void write_pvd_collection(const std::string& pvd_filename,
                                 const std::string& vtk_base_name,
                                 int num_timesteps,
                                 int save_interval,
                                 float dt) {
    std::ofstream pvd(pvd_filename);

    pvd << "<?xml version=\"1.0\"?>\n";
    pvd << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    pvd << "  <Collection>\n";

    for (int t = 0; t <= num_timesteps; t += save_interval) {
        char vti_file[256];
        std::snprintf(vti_file, sizeof(vti_file), "%s_%04d.vti", vtk_base_name.c_str(), t);
        pvd << "    <DataSet timestep=\"" << (t * dt) << "\" group=\"\" part=\"0\" file=\"" << vti_file << "\"/>\n";
    }

    pvd << "  </Collection>\n";
    pvd << "</VTKFile>\n";
    pvd.close();
}

}  // namespace VTK
