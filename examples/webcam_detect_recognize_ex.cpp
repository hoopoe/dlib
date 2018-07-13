// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    

    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead 
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/dnn.h>
#include <unordered_map>
#include <json.hpp>

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
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
    alevel0<
    alevel1<
    alevel2<
    alevel3<
    alevel4<
    max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
    input_rgb_image_sized<150>
    >>>>>>>>>>>>;

float FACE_RECOGNIZE_THRESH = 0.5; //todo: required to evaluate. 0.6 by default
// ----------------------------------------------------------------------------------------

//todo: remove dup code. TBD
void recognize(string facedb_path) {
    typedef matrix<float, 0, 1> face_vector;
    std::ifstream ifs(facedb_path);
    json facedb = json::parse(ifs);
    std::vector<string> names;
    std::vector<face_vector> vectors;
    
    for (auto &t : facedb["persons"]) {
        string name = t["name"];
        for (auto &j : t["vector"]) {
            std::vector<float> tmp = j;
            auto t1 = mat(tmp);
            auto t2 = matrix<float, 0, 1>(t1);
            names.push_back(name);
            vectors.push_back(t2);
            
        }
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "Unable to connect to camera" << endl;
        return;
    }

    image_window win;

    typedef matrix<float, 0, 1> face_vector;
    std::vector<std::vector<float>> facedb_tmp;

    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    while (!win.is_closed())
    {
        cv::Mat temp;
        if (!cap.read(temp))
        {
            break;
        }
        
        cv_image<bgr_pixel> cimg(temp);

        std::vector<rectangle> detected_faces = detector(cimg);

        if (detected_faces.size() == 0) {
            cout << "No faces found in image!" << endl;
            continue;
        }

        if (detected_faces.size() > 1) {
            cout << "Please train using only 1 face" << endl;
        }

        std::vector<matrix<rgb_pixel>> faces;
        for (auto face : detected_faces)
        {
            auto shape = sp(cimg, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip);
            faces.push_back(move(face_chip));
            win.clear_overlay();
            win.set_image(cimg);
            win.add_overlay(face);
        }

        std::vector<face_vector> face_descriptors = net(faces);
        face_vector face_descriptor = face_descriptors.front();

        bool recognized = false;
        int cx = 0;
        for (auto i : vectors) 
        {
            float dist = length(i - face_descriptor);
            if (dist < FACE_RECOGNIZE_THRESH) {
                recognized = true;
                break;
            }
            cx++;
        }
        if (recognized) {
            cout << "recognized " << names[cx] << endl;
        }
        else {
            cout << "NOT recognized " << endl;
        }
    }
}

void train(string name, string facedb_path) {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return ;
        }

        image_window win;

        typedef matrix<float, 0, 1> face_vector;
        //unordered_map<string, std::vector<face_vector>> facedb;
        //json facedb;
        std::vector<std::vector<float>> facedb_tmp;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor sp;
        deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
        // And finally we load the DNN responsible for face recognition.
        anet_type net;
        deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

        // Grab and process frames until the main window is closed by the user.
        while (!win.is_closed())
        {
            // Grab a frame
            cv::Mat temp;
            if (!cap.read(temp))
            {
                break;
            }
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<bgr_pixel> cimg(temp);

            // Detect faces 
            std::vector<rectangle> detected_faces = detector(cimg);

            if (detected_faces.size() == 0) {
                cout << "No faces found in image!" << endl;
                continue;
            }

            if (detected_faces.size() > 1) {
                cout << "Please train using only 1 face" << endl;
            }

            // Run the face detector on the image of our action heroes, and for each face extract a
            // copy that has been normalized to 150x150 pixels in size and appropriately rotated
            // and centered.
            std::vector<matrix<rgb_pixel>> faces;
            for (auto face : detected_faces)
            {
                auto shape = sp(cimg, face);
                matrix<rgb_pixel> face_chip;
                extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip);
                faces.push_back(move(face_chip));
                // Also put some boxes on the faces so we can see that the detector is finding
                // them.
                win.clear_overlay();
                win.set_image(cimg);
                win.add_overlay(face);
            }

            // This call asks the DNN to convert each face image in faces into a 128D vector.
            // In this 128D vector space, images from the same person will be close to each other
            // but vectors from different people will be far apart.  So we can use these vectors to
            // identify if a pair of images are from the same person or from different people.  
            std::vector<face_vector> face_descriptors = net(faces);
            face_vector face_descriptor = face_descriptors.front();

            std::vector<float> x(face_descriptor.begin(), face_descriptor.end());

            facedb_tmp.push_back(x);
        }

        json facedb;
        facedb["persons"] = std::vector<json>();
        json t;
        t["name"] = name;
        t["vector"] = facedb_tmp;

        facedb["persons"].push_back(t);

        std::ofstream o(facedb_path);
        o << std::setw(4) << facedb << std::endl;
}

int main(int argc, char** argv)
{
    if (argc != 2 && argc != 3)
    {
        cout << argc << endl;
        cout << "Run this example for training by invoking it like this: " << endl;
        cout << "   ./webcam_detect_recognize_ex <person name> <db name>" << endl;
        cout << "Run this example for recognition by invoking it like this: " << endl;
        cout << "   ./webcam_detect_recognize_ex <db name>" << endl;
        cout << endl;
        cout << "You will also need to get the face landmarking model file as well as " << endl;
        cout << "the face recognition model file.  Download and then decompress these files from: " << endl;
        cout << "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2" << endl;
        cout << "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2" << endl;
        cout << endl;
        return 1;
    }

    if (argc == 2) {// recognition
        string facedb_path = argv[1];
        recognize(facedb_path);
    }
    else if (argc == 3) {// training
        string name = argv[1];
        string facedb_path = argv[2];
        try
        {
            train(name, facedb_path);
        }
        catch (serialization_error& e)
        {
            cout << "You need dlib's default face landmarking model file to run this example." << endl;
            cout << "You can get it from the following URL: " << endl;
            cout << "   http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2" << endl;
            cout << endl << e.what() << endl;
        }
        catch (exception& e)
        {
            cout << e.what() << endl;
        }
    }
}

