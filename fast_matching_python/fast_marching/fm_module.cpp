#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "calc_geodesics.h"
#include <pybind11/embed.h>
#include <signal.h>

namespace py = pybind11;



void signalHandler(int signum) {
    exit(signum);
}


// vector<vector<double> > fast_marching(string data_dir, bool verbose) {
//     vector<vector<double> > res = calc_geodesic_matrix(data_dir, verbose);
//     return res;
// }


// PYBIND11_MODULE(fast_marching, m) {
//     py::bind_vector<vector<vector<double> > > (m, "FloatVector2D");
//     m.doc() = "fast marching for mesh geodesics"; 

//     signal(SIGINT, signalHandler);

//     m.def("fast_marching", &fast_marching, "calculate pairwise geodesics of a mesh", 
//           py::arg("data_dir"), py::arg("verbose"));
// }







/*******************************************************************************************
    Omer:
*******************************************************************************************/
#include "calc_geodesics.h"
#include "che_off.h"
#include "progress_bar.h"




std::vector<double> fast_marching_single(std::string data_dir, int source_index, bool verbose) {
    che *mesh = new che_off(data_dir);
    vector<double> res;
    size_t n_vertices = mesh->n_vertices();
    progressbar bar(n_vertices);

    

    if (verbose) { bar.update(); }

    auto *toplesets = new index_t[n_vertices];
    auto *sorted_index = new index_t[n_vertices];
    vector<index_t> limits;
    vector<index_t> source = {static_cast<unsigned int>(source_index)};

    mesh->compute_toplesets(toplesets, sorted_index, limits, source);
    res = fast_marching_single_vert(mesh, source);

    delete[] toplesets;
    delete[] sorted_index;

    delete mesh;
    return res;
}


PYBIND11_MODULE(fast_marching, m) {
    py::bind_vector<std::vector<std::vector<double>>>(m, "FloatVector2D");
    py::bind_vector<std::vector<double>>(m, "FloatVector1D");

    m.doc() = "fast marching for mesh geodesics";

    signal(SIGINT, signalHandler);

    m.def("fast_marching", &fast_marching_single, "calculate geodesics from a single source index",
          py::arg("data_dir"), py::arg("source_index"), py::arg("verbose"));
}
