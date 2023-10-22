#include <iostream>
#include <vector>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Dense>
#include <igl/writeOFF.h>
#include <fstream>
using namespace std;

struct RadialCoordinates {
    double r;     // Radius
    double theta; // Azimuthal angle
    double phi;   // Polar angle
};

void saveMeshOFF(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const string &filename) {
    if (igl::writeOFF(filename, V, F)) {
        cout << "Mesh saved to " << filename << endl;
    } else {
        cerr << "Error saving the mesh to " << filename << endl;
    }
}

vector<Eigen::Vector3d> eigenToVector(const Eigen::MatrixXd &epoints) {
    vector<Eigen::Vector3d> vpoints;
    for (int i = 0; i < epoints.rows(); ++i) {
        Eigen::Vector3d point(epoints(i, 0), epoints(i, 1), epoints(i, 2));
        vpoints.push_back(point);
    }
    return vpoints;
}

Eigen::MatrixXd vectorToEigen(const vector<Eigen::Vector3d> &vpoints) {
    int numPoints = vpoints.size();

    if (numPoints == 0) {
        // Return an empty matrix if the vector is empty
        return Eigen::MatrixXd();
    }

    Eigen::MatrixXd epoints(numPoints, 3);

    for (int i = 0; i < numPoints; ++i) {
        epoints(i, 0) = vpoints[i](0); // x-coordinate
        epoints(i, 1) = vpoints[i](1); // y-coordinate
        epoints(i, 2) = vpoints[i](2); // z-coordinate
    }

    return epoints;
}

Eigen::Vector3d getBarycenter(const vector<Eigen::Vector3d> &points) {
    Eigen::Vector3d barycenter(0.0, 0.0, 0.0);

    for (const Eigen::Vector3d &point : points) {
        barycenter += point;
    }

    if (!points.empty()) {
        barycenter /= points.size();
    }

    return barycenter;
}

RadialCoordinates cartesianToRadial(const Eigen::Vector3d &cartesian, const Eigen::Vector3d &barycenter) {
    RadialCoordinates radial;
    radial.r = (cartesian - barycenter).norm();
    radial.theta = atan2(cartesian(1) - barycenter(1), cartesian(0) - barycenter(0));
    radial.phi = acos((cartesian(2) - barycenter(2)) / radial.r);
    return radial;
}

void sphericalToCartesian(double r, double theta, double phi, Eigen::Vector3d &cartesian, const Eigen::Vector3d &barycenter) {
    cartesian(0) = r * sin(phi) * cos(theta) + barycenter(0);
    cartesian(1) = r * sin(phi) * sin(theta) + barycenter(1);
    cartesian(2) = r * cos(phi) + barycenter(2);
}

RadialCoordinates getMinRadialFromBin(const vector<RadialCoordinates> &radials, const vector<int> &bin) {
    if (bin.empty()) {
        cerr << "The bin is empty." << endl;
        return RadialCoordinates{0.0, 0.0, 0.0}; // Default values
    }

    RadialCoordinates minRadial = radials[bin[0]];
    for (unsigned int index : bin) {
        if (radials[index].r < minRadial.r) {
            minRadial = radials[index];
        }
    }

    return minRadial;
}

RadialCoordinates getMaxRadialFromBin(const vector<RadialCoordinates> &radials, const vector<int> &bin) {
    if (bin.empty()) {
        cerr << "The bin is empty." << endl;
        return RadialCoordinates{0.0, 0.0, 0.0}; // Default values
    }

    RadialCoordinates maxRadial = radials[bin[0]];
    for (unsigned int index : bin) {
        if (radials[index].r > maxRadial.r) {
            maxRadial = radials[index];
        }
    }

    return maxRadial;
}

void binToPowerOf(std::vector<RadialCoordinates> &radials, std::vector< int> bin, double k) {
    for (unsigned int index : bin) {
        radials[index].r = pow(radials[index].r, k);
    }
}

RadialCoordinates getMeanOfBin(const std::vector<RadialCoordinates> &radials, const std::vector< int> &bin) {
    RadialCoordinates meanRadial = {0.0, 0.0, 0.0};
    double count = 0.0;

    for (unsigned int index : bin) {
        meanRadial.r += radials[index].r;
        meanRadial.theta += radials[index].theta;
        meanRadial.phi += radials[index].phi;
        count += 1.0;
    }

    if (count > 0) {
        meanRadial.r /= count;
        meanRadial.theta /= count;
        meanRadial.phi /= count;
    }

    return meanRadial;
}

void normalizeRadials(vector<RadialCoordinates> &radials, const RadialCoordinates &minVals, const RadialCoordinates &maxVals) {
    for (RadialCoordinates &radial : radials) {
        radial.r = (radial.r - minVals.r) / (maxVals.r - minVals.r);
        radial.theta = (radial.theta - minVals.theta) / (maxVals.theta - minVals.theta);
        radial.phi = (radial.phi - minVals.phi) / (maxVals.phi - minVals.phi);
    }
}

void denormalizeRadials(vector<RadialCoordinates> &radials, const RadialCoordinates &minVals, const RadialCoordinates &maxVals) {
    for (RadialCoordinates &radial : radials) {
        radial.r = radial.r * (maxVals.r - minVals.r) + minVals.r;
        radial.theta = radial.theta * (maxVals.theta - minVals.theta) + minVals.theta;
        radial.phi = radial.phi * (maxVals.phi - minVals.phi) + minVals.phi;
    }
}

void saveHistogram(const vector<int> &histo, const char *path, double minValueBins, double binWidth) {
    ofstream flux(path);
    for (int i = 0; i < histo.size(); i++) {
        double x = minValueBins + i * binWidth;
        flux << x << " " << histo[i] << endl;
    }
}

std::vector<std::vector< int>> histogram(const std::vector<RadialCoordinates> &radials, int k) {
    std::vector<std::vector< int>> histogram(k);

    double minRadial = std::numeric_limits<double>::max();
    double maxRadial = std::numeric_limits<double>::min();

    for (const RadialCoordinates &radial : radials) {
        if (radial.r < minRadial) {
            minRadial = radial.r;
        }
        if (radial.r > maxRadial) {
            maxRadial = radial.r;
        }
    }

    for (int i = 0; i < k; ++i) {
        double Bmin = minRadial + ((maxRadial - minRadial) / k) * i;
        double Bmax = minRadial + ((maxRadial - minRadial) / k) * (i + 1);

        for (unsigned int j = 0; j < radials.size(); ++j) {
            if (radials[j].r > Bmin && radials[j].r <= Bmax) {
                histogram[i].push_back(j);
            }
        }
    }

    return histogram;
}

vector<bool> stringToBinary(const string &input) {
    vector<bool> bv;

    for (char character : input) {
        for (int i = 7; i >= 0; --i) {
            bv.push_back((character >> i) & 1);
        }
    }

    return bv;
}

string binaryToString(const vector<bool> &bv) {
    string result;

    for (size_t i = 0; i < bv.size(); i += 8) {
        char character = 0;
        for (int j = 0; j < 8; ++j) {
            if (i + j < bv.size()) {
                character |= bv[i + j] << (7 - j);
            }
        }
        result.push_back(character);
    }

    return result;
}

void embedMessage(vector<RadialCoordinates> &radials, const vector<bool> &message, int n, double alpha) {
      vector<vector< int>> bins = histogram(radials, n);
    double deltaK = 0.1;

    for (int i = 0; i < n; ++i) {
        RadialCoordinates minRadial = getMinRadialFromBin(radials, bins[i]);
        RadialCoordinates maxRadial = getMaxRadialFromBin(radials, bins[i]);
        normalizeRadials(radials, minRadial, maxRadial);

        double kCopy = 0.5;
        RadialCoordinates mean = getMeanOfBin(radials, bins[i]);

        if (message[i]) {
            while (mean.r < (0.5 + alpha)) {
                kCopy -= deltaK;
                binToPowerOf(radials, bins[i], kCopy);
                mean = getMeanOfBin(radials, bins[i]);
            }
        } else {
            while (mean.r > (0.5 - alpha)) {
                kCopy += deltaK;
                binToPowerOf(radials, bins[i], kCopy);
                mean = getMeanOfBin(radials, bins[i]);
            }
        }

        denormalizeRadials(radials, minRadial, maxRadial);
    }
}

vector<bool> extractMessage(vector<RadialCoordinates> &radials,  int n, double alpha) {
    vector<vector< int>> bins = histogram(radials, n);
    double deltaK = 0.1;
    vector<bool> extractedMessage;
    for (int i = 0; i < n; ++i) {
        RadialCoordinates minRadial = getMinRadialFromBin(radials, bins[i]);
        RadialCoordinates maxRadial = getMaxRadialFromBin(radials, bins[i]);
        normalizeRadials(radials, minRadial, maxRadial);

        double kCopy = 0.5;
        RadialCoordinates mean = getMeanOfBin(radials, bins[i]);

        bool bit = false;
        if (mean.r > (0.5 + alpha)) {
            bit = true;
        }

        extractedMessage.push_back(bit);
        denormalizeRadials(radials, minRadial, maxRadial);
    }

    return extractedMessage;
}

int main(int argc, char *argv[]) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    vector<Eigen::Vector3d> points;

    igl::readOFF("../models/avion_n.off", V, F);

    points = eigenToVector(V);

    // Rest of your code for visualization, if needed
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);

    Eigen::Vector3d barycenter = getBarycenter(points);

    // Convert Cartesian coordinates to radial coordinates
    vector<RadialCoordinates> radials;
    for (const Eigen::Vector3d &point : points) {
        radials.push_back(cartesianToRadial(point, barycenter));
    }

    string messageToEmbed = "Bonjour";
    vector<bool> binaryMessageToEmbed = stringToBinary(messageToEmbed);

    double alpha = 0.05; // Embedding strength

    // Call the function to embed the message
    embedMessage(radials, binaryMessageToEmbed, binaryMessageToEmbed.size(), alpha);

    // Call the function to extract the message
    vector<bool> extractedMessage = extractMessage(radials, binaryMessageToEmbed.size(), alpha);

    // Convert the extracted message to text
    string reconstructedMessage = binaryToString(extractedMessage);
    cout << "Extracted message: " << reconstructedMessage << endl;

    vector<Eigen::Vector3d> modifiedPoints;
    for (const RadialCoordinates &radial : radials) {
        Eigen::Vector3d cartesian;
        sphericalToCartesian(radial.r, radial.theta, radial.phi, cartesian, barycenter);
        modifiedPoints.push_back(cartesian);
    }

    // Call the function to save the mesh in OFF format
    Eigen::MatrixXd modifiedV = vectorToEigen(modifiedPoints);

    if (igl::writeOFF("tatooed.off", modifiedV, F)) {
        cout << "Modified mesh saved as 'tatooed.off'" << endl;
    } else {
        cerr << "Error saving the modified mesh." << endl;
    }

    viewer.launch();

    return 0;
}
