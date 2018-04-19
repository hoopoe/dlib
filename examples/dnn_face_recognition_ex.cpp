// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the dlib C++
    Library.  In it, we will show how to do face recognition.  This example uses the
    pretrained dlib_face_recognition_resnet_model_v1 model which is freely available from
    the dlib web site.  This model has a 99.38% accuracy on the standard LFW face
    recognition benchmark, which is comparable to other state-of-the-art methods for face
    recognition as of February 2017. 
    
    In this example, we will use dlib to do face clustering.  Included in the examples
    folder is an image, bald_guys.jpg, which contains a bunch of photos of action movie
    stars Vin Diesel, The Rock, Jason Statham, and Bruce Willis.   We will use dlib to
    automatically find their faces in the image and then to automatically determine how
    many people there are (4 in this case) as well as which faces belong to each person.
    
    Finally, this example uses a network with the loss_metric loss.  Therefore, if you want
    to learn how to train your own models, or to get a general introduction to this loss
    layer, you should read the dnn_metric_learning_ex.cpp and
    dnn_metric_learning_on_images_ex.cpp examples.
*/

#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <json.hpp>
#include <experimental/filesystem>
#include <dlib/image_saver/save_png.h>
#include <dlib/svm/svm_multiclass_linear_trainer.h>

using namespace dlib;
using namespace std;
using json = nlohmann::json;
// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
);

// ----------------------------------------------------------------------------------------

namespace filesys = std::experimental::filesystem;

std::vector<std::string> getAllFilesInDir(const std::string &dirPath, const std::vector<std::string> acceptExt = {})
{

    // Create a vector of string
    std::vector<std::string> listOfFiles;
    try {
        // Check if given path exists and points to a directory
        if (filesys::exists(dirPath) && filesys::is_directory(dirPath))
        {
            // Create a Recursive Directory Iterator object and points to the starting of directory
            filesys::recursive_directory_iterator iter(dirPath);

            // Create a Recursive Directory Iterator object pointing to end.
            filesys::recursive_directory_iterator end;

            // Iterate till end
            while (iter != end)
            {
                // Check if current entry is a directory and if exists in skip list
                if (filesys::is_regular_file(iter->path()) && 
                    (std::find(acceptExt.begin(), acceptExt.end(), iter->path().extension()) != acceptExt.end()))
                {
                    listOfFiles.push_back(iter->path().string());
                }
                error_code ec;
                // Increment the iterator to point to next entry in recursive iteration
                iter.increment(ec);
                if (ec) {
                    std::cerr << "Error While Accessing : " << iter->path().string() << " :: " << ec.message() << '\n';
                }
            }
        }
    }
    catch (std::system_error & e)
    {
        std::cerr << "Exception :: " << e.what();
    }
    return listOfFiles;
}

void buildClusters(std::string sourceDir, std::string chippedDir)
{
    // The first thing we are going to do is load all our models.  First, since we need to
    // find faces in the image we will need a face detector:
    frontal_face_detector detector = get_frontal_face_detector();
    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
    // And finally we load the DNN responsible for face recognition.
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    std::vector<std::string> listOfFiles = getAllFilesInDir(sourceDir, { ".jpg", ".JPG", ".png", ".PNG" });
    std::vector<matrix<rgb_pixel>> faces;

    for (auto str : listOfFiles) {
        std::cout << str << std::endl;
        matrix<rgb_pixel> img;
        load_image(img, str);

        for (auto face : detector(img))
        {
            auto shape = sp(img, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
            faces.push_back(move(face_chip));
            // Also put some boxes on the faces so we can see that the detector is finding
            // them.
            //win.add_overlay(face);
            //std::string filename = chippedDir + "/" + cast_to_string(cluster_id) + ".png";
        }
    }
    if (faces.size() == 0)
    {
        cout << "No faces found in image!" << endl;
        //return 1;
    }

    std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);

    // In particular, one simple thing we can do is face clustering.  This next bit of code
    // creates a graph of connected faces and then uses the Chinese whispers graph clustering
    // algorithm to identify how many people there are and which faces belong to whom.
    std::vector<sample_pair> edges;
    for (size_t i = 0; i < face_descriptors.size(); ++i)
    {
        for (size_t j = i; j < face_descriptors.size(); ++j)
        {
            // Faces are connected in the graph if they are close enough.  Here we check if
            // the distance between two face descriptors is less than 0.6, which is the
            // decision threshold the network was trained to use.  Although you can
            // certainly use any other threshold you find useful.
            if (length(face_descriptors[i] - face_descriptors[j]) < 0.5) {
                edges.push_back(sample_pair(i, j));
            }
        }
    }
    std::vector<unsigned long> labels;
    const auto num_clusters = chinese_whispers(edges, labels);
    // This will correctly indicate that there are 4 people in the image.
    cout << "number of people found in the image: " << num_clusters << endl;


    // Now let's display the face clustering results on the screen.  You will see that it
    // correctly grouped all the faces. 
    //std::vector<image_window> win_clusters(num_clusters);
    for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
    {
        std::vector<matrix<rgb_pixel>> temp;
        for (size_t j = 0; j < labels.size(); ++j)
        {
            if (cluster_id == labels[j]) {
                temp.push_back(faces[j]);
            }
        }
        //save_png(temp, cast_to_string(cluster_id));
        std::string filename = chippedDir + "/" + cast_to_string(cluster_id) + ".png";
        save_png(tile_images(temp), filename);
        /*win_clusters[cluster_id].set_title("face cluster " + cast_to_string(cluster_id));
        win_clusters[cluster_id].set_image(tile_images(temp));*/
    }

    //randomize_samples(samples, labels);
    //cout << "cross validation: \n" << cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;

    //// Finally, let's print one of the face descriptors to the screen.  
    //cout << "face descriptor for one face: " << trans(face_descriptors[0]) << endl;

    //// It should also be noted that face recognition accuracy can be improved if jittering
    //// is used when creating face descriptors.  In particular, to get 99.38% on the LFW
    //// benchmark you need to use the jitter_image() routine to compute the descriptors,
    //// like so:
    //matrix<float, 0, 1> face_descriptor = mean(mat(net(jitter_image(faces[0]))));
    //cout << "jittered face descriptor for one face: " << trans(face_descriptor) << endl;
    //// If you use the model without jittering, as we did when clustering the bald guys, it
    //// gets an accuracy of 99.13% on the LFW benchmark.  So jittering makes the whole
    //// procedure a little more accurate but makes face descriptor calculation slower.


    cout << "hit enter to terminate" << endl;
    cin.get();
}

std::string remove_extension(const std::string& filename) {
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos) return filename;
    return filename.substr(0, lastdot);
}

void dumpClustersToJson(std::string chippedDir)
{
    // The first thing we are going to do is load all our models.  First, since we need to
    // find faces in the image we will need a face detector:
    frontal_face_detector detector = get_frontal_face_detector();
    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
    // And finally we load the DNN responsible for face recognition.
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    std::vector<std::string> listOfFiles = getAllFilesInDir(chippedDir, { ".jpg", ".JPG", ".png", ".PNG" });
    std::vector<matrix<rgb_pixel>> faces;

    json j;
    for (auto filepath : listOfFiles) {
        std::cout << filepath << std::endl;

        matrix<rgb_pixel> img;
        load_image(img, filepath);

        for (auto face : detector(img))
        {
            auto shape = sp(img, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
            faces.push_back(move(face_chip));
        }

        std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
        json cluster;
        std::string filename = filesys::path(filepath).filename().string();
        cluster["filename"] = filename;
        cluster["name"] = remove_extension(filename);
        for (size_t i = 0; i < face_descriptors.size(); ++i)
        {
            cluster[cast_to_string(i)] = face_descriptors[i];
            serialize("fvector_" + std::to_string(i)) << face_descriptors[i];
        }
        j[filename] = cluster;
    }
    if (faces.size() == 0)
    {
        cout << "No faces found in image!" << endl;
    }
    std::ofstream o(chippedDir + "/" + "profiles.json");
    o << std::setw(4) << j << std::endl;
}

void dumpClustersToSVM(std::string chippedDir)
{
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    std::vector<std::string> listOfFiles = getAllFilesInDir(chippedDir, { ".jpg", ".JPG", ".png", ".PNG" });
    

    typedef matrix<float, 0, 1> sample_type;
    std::vector<sample_type> samples;
    std::vector<string> labels;

    for (auto filepath : listOfFiles) {
        std::cout << filepath << std::endl;
        std::string filename = filesys::path(filepath).filename().string();

        matrix<rgb_pixel> img;
        load_image(img, filepath);

        std::vector<matrix<rgb_pixel>> faces;
        for (auto face : detector(img))
        {
            auto shape = sp(img, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
            faces.push_back(move(face_chip));
        }

        std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
        for (size_t i = 0; i < face_descriptors.size(); ++i)
        {
            labels.push_back(filename);
            samples.push_back(face_descriptors[i]);
        }

        cout << "size: " << labels.size() << endl;
    }
    /*if (faces.size() == 0)
    {
        cout << "No faces found in image!" << endl;
    }*/

    typedef linear_kernel<sample_type> lin_kernel;

    // Define the SVM multiclass trainer
    typedef svm_multiclass_linear_trainer <lin_kernel, string> svm_mc_trainer;
    svm_mc_trainer trainer;
    trainer.set_c(1);

    multiclass_linear_decision_function<lin_kernel, string> df = trainer.train(samples, labels);

    serialize("faces_linear.svm") << df;

    cout << "First run: " << endl;
    for (int i = 0; i < labels.size(); i++) {
        std::pair<string, float> res = df.predict(samples[i]);
        cout << labels[i] << " : " << res.first << " : " << res.second << endl;
    }

    multiclass_linear_decision_function<lin_kernel, string> newdf;
    deserialize("faces_linear.svm") >> newdf;

    cout << "Second run: " << endl;
    for (int i = 0; i < labels.size(); i++) {
        std::pair<string, float> res = df.predict(samples[i]);
        cout << labels[i] << " : " << res.first << " : " << res.second << endl;
    }
}

void performanceTest(std::string dir)
{
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    typedef matrix<float, 0, 1> sample_type;
    typedef linear_kernel<sample_type> lin_kernel;
    multiclass_linear_decision_function<lin_kernel, string> newdf;
    deserialize("faces_linear.svm") >> newdf;

    std::vector<std::string> listOfFiles = getAllFilesInDir(dir, { ".jpg", ".JPG", ".png", ".PNG" });

    std::vector<sample_type> samples;
    std::vector<string> labels;

    for (auto filepath : listOfFiles) {
        std::string filename = filesys::path(filepath).filename().string();
        ofstream f;
        f.open(filename + ".perf");

        matrix<rgb_pixel> img;
        load_image(img, filepath);

        std::vector<matrix<rgb_pixel>> faces;

        auto start = std::chrono::high_resolution_clock::now();
        auto dets = detector(img);
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> detection_elapsed = finish - start;
        f << "detection_elapsed: " << detection_elapsed.count() << "\n";

        start = std::chrono::high_resolution_clock::now();
        for (auto face : dets)
        {
            auto shape = sp(img, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
            faces.push_back(move(face_chip));
        }

        finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> face_chip_elapsed = finish - start;
        f << "face_chip_elapsed: " << face_chip_elapsed.count() << "\n";

        start = std::chrono::high_resolution_clock::now();
        std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
        finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> feature_elapsed = finish - start;
        f << "feature_elapsed: " << feature_elapsed.count() << "\n";
        for (size_t i = 0; i < face_descriptors.size(); ++i)
        {
            labels.push_back(filename);
            samples.push_back(face_descriptors[i]);
        }

        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < labels.size(); i++) {
            std::pair<string, float> res = newdf.predict(samples[i]);
        }
        finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> svm_elapsed = finish - start;
        f << "svm_elapsed: " << svm_elapsed.count() << "\n";
        f.close();
    }
}

void dumpFaceFeatureVector(std::string filepath)
{
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    std::string filename = filesys::path(filepath).filename().string();

    size_t lastindex = filename.find_last_of(".");
    string pure_filename = filename.substr(0, lastindex);

    matrix<rgb_pixel> img;
    load_image(img, filepath);

    std::vector<matrix<rgb_pixel>> faces;
    for (auto face : detector(img))
    {
        auto shape = sp(img, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
        faces.push_back(move(face_chip));
    }

    std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
    for (size_t i = 0; i < face_descriptors.size(); ++i)
    {
        serialize(pure_filename + std::to_string(i) + ".vec") << face_descriptors[i];
    }
}

typedef matrix<double, 3, 1> sample_type;
int main(int argc, char** argv)
{
    if (argc == 2)
    {
        //dumpClustersToJson(argv[1]);
        //dumpClustersToSVM(argv[1]);
        //performanceTest(argv[1]);
        dumpFaceFeatureVector(argv[1]);
    }
    else {
        if (argc == 3)
        {
            buildClusters(argv[1], argv[2]);
        }
        else {
            cout << "Run this example by invoking it like this: " << endl;
            cout << "   ./dnn_face_recognition_ex faces/bald_guys.jpg" << endl;
            cout << endl;
            cout << "You will also need to get the face landmarking model file as well as " << endl;
            cout << "the face recognition model file.  Download and then decompress these files from: " << endl;
            cout << "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2" << endl;
            cout << "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2" << endl;
            cout << endl;
            return 1;
        }
    }

    
}


// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
)
{
    // All this function does is make 100 copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently. They are also randomly
    // mirrored left to right.
    thread_local dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> crops; 
    for (int i = 0; i < 100; ++i)
        crops.push_back(jitter_image(img,rnd));

    return crops;
}

// ----------------------------------------------------------------------------------------

