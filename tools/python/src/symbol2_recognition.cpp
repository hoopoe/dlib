// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/matrix.h>
#include <dlib/geometry/vector.h>
#include <dlib/dnn.h>
#include <dlib/image_transforms.h>
#include "indexing.h"
#include <dlib/image_io.h>
#include <dlib/clustering.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

using namespace dlib;
using namespace std;

namespace py = pybind11;

class symbol2_recognition_model_v1
{

public:

    symbol2_recognition_model_v1(const std::string& model_filename)
    {
        deserialize(model_filename) >> net;
    }

    int detect (
        py::array pyimage
    )
    {
        matrix<unsigned char> image;
        if (is_image<unsigned char>(pyimage))
            assign_image(image, numpy_image<unsigned char>(pyimage));
        std::vector<matrix<unsigned char>> training_images;        
        training_images.push_back(image);

        std::vector<unsigned long> predicted_labels = net(training_images);
        return predicted_labels.front();
    }

private:

    using net_type = loss_multiclass_log<
        fc<15,
        relu<fc<84,
        relu<fc<120,
        max_pool<2,2,2,2,relu<con<16,5,5,1,1,
        max_pool<2,2,2,2,relu<con<6,5,5,1,1,
        input<matrix<unsigned char>> 
        >>>>>>>>>>>>;

    net_type net;
};


void bind_symbol2_recognition(py::module &m)
{
    {
    py::class_<symbol2_recognition_model_v1>(m, "symbol2_recognition_model_v1", "This object maps some symbols BigA BigC BigD BigE BigF BigG BigI BigM BigN BigO BigP BigR BigT BigV BigX into digits 0..14")
        .def(py::init<std::string>())
        .def(
            "__call__", 
            &symbol2_recognition_model_v1::detect, 
            py::arg("img"),
            "Detect symbol number for image"
            );
    }
}

