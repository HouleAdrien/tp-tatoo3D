#include <iostream>
#include <vector>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Dense>
#include <igl/writeOFF.h>
using namespace std;

struct RadialCoordinates {
    double r;     // Rayon
    double theta; // Angle azimutal
    double phi;   // Angle polaire
};

void saveMeshOFF(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const string &filename) {
    if (igl::writeOFF(filename, V, F)) { 
        cout << "Maillage sauvegardé dans " << filename << endl;
    } else {
        cerr << "Erreur lors de la sauvegarde du maillage dans " << filename << endl;
    }
}

void saveHistogram(const vector<int>& histo,char* path)
{
    ofstream flux(path);
    for(int i = 0; i < histo.size();i++)    {    flux << (i/(float)histo.size())  << " " << histo[i] << endl;}
}

vector<Eigen::Vector3d> eigenToVector(const Eigen::MatrixXd &epoints) {
    vector<Eigen::Vector3d> vpoints;
    for (int i = 0; i < epoints.rows(); ++i) 
    {
        Eigen::Vector3d point(epoints(i, 0), epoints(i, 1), epoints(i, 2));
        vpoints.push_back(point);
    }
    return vpoints;
}

Eigen::MatrixXd vectorToEigen(const vector<Eigen::Vector3d>& vpoints) {
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

RadialCoordinates cartesianToRadial(const Eigen::Vector3d &cartesian) {
    RadialCoordinates radial;
    radial.r = cartesian.norm();
    radial.theta = atan2(cartesian(1), cartesian(0));
    radial.phi = acos(cartesian(2) / radial.r);
    return radial;
}

void sphericalToCartesian(double r, double theta, double phi, Eigen::Vector3d &cartesian) {
    cartesian(0) = r * sin(phi) * cos(theta);
    cartesian(1) = r * sin(phi) * sin(theta);
    cartesian(2) = r * cos(phi);
}

void findMinMaxRadials(const vector<RadialCoordinates> &radials, RadialCoordinates &minRadial, RadialCoordinates &maxRadial) {
    if (radials.empty()) {
        minRadial = {0.0, 0.0, 0.0};
        maxRadial = {0.0, 0.0, 0.0};
        return;
    }

    minRadial = radials[0];
    maxRadial = radials[0];

    for (const RadialCoordinates &radial : radials) {
        if (radial.r < minRadial.r) {
            minRadial = radial;
        }
        if (radial.r > maxRadial.r) {
            maxRadial = radial;
        }
    }
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

vector<int> makeBinsHisto(int binsNumber, double minValueBins, double maxValueBins, vector<RadialCoordinates> &radials) {
    vector<int> histogram(binsNumber, 0);

    // Calcul de la plage de valeurs
    double range = maxValueBins - minValueBins;
    if (range <= 0.0) {
        cerr << "La plage de valeurs est incorrecte." << endl;
        return histogram;
    }

    // Largeur de chaque bin
    double binWidth = range / binsNumber;

    // Remplissage de l'histogramme
    for (const RadialCoordinates &radial : radials) {
        // Calcul de l'indice du bin
        int binIndex = static_cast<int>((radial.r - minValueBins) / binWidth);

        // Vérification des limites
        if (binIndex < 0) {
            binIndex = 0;
        } else if (binIndex >= binsNumber) {
            binIndex = binsNumber - 1;
        }

        // Incrémentation du bin correspondant
        histogram[binIndex]++;
    }

    return histogram;
}

vector<bool> stringToBinary(const string& input) 
{
    vector<bool> bv;
    
    for (char character : input) 
    {
        for (int i = 7; i >= 0; --i) 
        {
            bv.push_back((character >> i) & 1);
        }
    }
    
    return bv;
}

string binaryToString(const vector<bool>& bv) 
{
    string result;
    
    for (size_t i = 0; i < bv.size(); i += 8) 
    {
        char character = 0;
        for (int j = 0; j < 8; ++j) 
        {
            if (i + j < bv.size()) 
            {
                character |= bv[i + j] << (7 - j);
            }
        }
        result.push_back(character);
    }
    
    return result;
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
    //cout << "Barycentre du maillage : " << barycenter << endl;

    // Test conversion sphérique pour un point
   /* double r, theta, phi;

    Eigen::Vector3d cartesianPoint(1.0, 2.0, 3.0); // Exemple de coordonnées cartésiennes
    cartesianToSpherical(cartesianPoint, r, theta, phi);
    cout << "Coordonnées sphériques : r = " << r << ", theta = " << theta << ", phi = " << phi << endl;

    Eigen::Vector3d newCartesian;
    sphericalToCartesian(r, theta, phi, newCartesian); // Exemple de conversion de sphérique à cartésienne
    cout << "Coordonnées cartésiennes : " << newCartesian << endl; */

    // Convertir les coordonnées cartésiennes en coordonnées radiales
    vector<RadialCoordinates> radials;
    for (const Eigen::Vector3d &point : points) {
        radials.push_back(cartesianToRadial(point));
    }

    // Trouver les valeurs minimales et maximales des sommets radiaux
    RadialCoordinates minRadial, maxRadial;
    findMinMaxRadials(radials, minRadial, maxRadial);

    // Afficher les valeurs minimales et maximales des sommets radiaux
    //cout << "Valeur minimale des sommets radiaux - R: " << minRadial.r << ", Theta: " << minRadial.theta << ", Phi: " << minRadial.phi << endl;
   // cout << "Valeur maximale des sommets radiaux - R: " << maxRadial.r << ", Theta: " << maxRadial.theta << ", Phi: " << maxRadial.phi << endl;

  /*  cout << "Avant normalisation : "<< endl;
    for (int i = 0; i < min(5, static_cast<int>(radials.size())); ++i) {
        const RadialCoordinates &radial = radials[i];
        cout << "(r: " << radial.r << ", theta: " << radial.theta << ", phi: " << radial.phi << ") " << endl;
    }
  */

  normalizeRadials(radials,  minRadial, maxRadial);
    // Afficher les 5 premiers éléments après la normalisation
   /* cout << "Après normalisation : "<< endl;
    for (int i = 0; i < min(5, static_cast<int>(radials.size())); ++i) {
        const RadialCoordinates &radial = radials[i];
        cout << "(r: " << radial.r << ", theta: " << radial.theta << ", phi: " << radial.phi << ") "<< endl;
    } */

   
    // Sauvegardez les histogrammes dans des fichiers .dat
    
    saveHistogram(makeBinsHisto(10,0,1,radials), "bins10.dat");
    saveHistogram(makeBinsHisto(100,0,1,radials), "bins100.dat");
    saveHistogram(makeBinsHisto(1000,0,1,radials), "bins1000.dat");

    // Dénormalisation
    denormalizeRadials(radials, minRadial, maxRadial);

    string message = "Bonjour c'est un message";
    vector<bool> binaryMessage = stringToBinary(message);
    string reconstructedMessage = binaryToString(binaryMessage);
    cout << reconstructedMessage << endl;

    // Afficher les 5 premiers éléments après la dénormalisation
   /* cout << "Après dénormalisation : "<< endl;
    for (int i = 0; i < min(5, static_cast<int>(radials.size())); ++i) {
        const RadialCoordinates &radial = radials[i];
        cout << "(r: " << radial.r << ", theta: " << radial.theta << ", phi: " << radial.phi << ") "<< endl;
    } */


    // Appel de la fonction pour sauvegarder le maillage au format OFF
 //   saveMeshOFF(embeddedV, F, "test.off");

    viewer.launch();

    return 0;
}
